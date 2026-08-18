"""Parquet storage for routing traces.

Why columnar: Phase 2 records, per token per MoE layer, six expert IDs plus six weights plus
(subsampled) 128 fp16 logits. A 500-sequence x 512-token eval over 18 layers is ~4.6M rows
before logits. Parquet with predicate pushdown makes `metrics.py` queryable instead of a
memory problem, and lets analysis run on a laptop while the GPU node is released.

Schema (one row per token per layer)::

    condition     str      condition ID (A..I)
    language      str      language code
    layer         int32    MoE layer index (1..18 for sarvam-30b)
    seq_id        int32    sequence index within the eval set
    position      int32    token position within the sequence
    fingerprint   str      hash of the full token sequence — the teacher-forcing guard
    expert_ids    list<int16>    top-k chosen experts, in rank order
    expert_weights list<float32> their routing weights, same order
    logits        list<float16>  full router logits, or null on non-sampled positions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        pa.field("condition", pa.string()),
        pa.field("language", pa.string()),
        pa.field("layer", pa.int32()),
        pa.field("seq_id", pa.int32()),
        pa.field("position", pa.int32()),
        pa.field("fingerprint", pa.string()),
        pa.field("expert_ids", pa.list_(pa.int16())),
        pa.field("expert_weights", pa.list_(pa.float32())),
        pa.field("logits", pa.list_(pa.float16())),
    ]
)

ROWS_PER_FLUSH = 250_000


class RoutingTraceWriter:
    """Buffered Parquet writer for one (condition, language) pair."""

    def __init__(self, out_dir: Path, condition: str, language: str):
        self.out_dir = Path(out_dir) / f"condition={condition}" / f"language={language}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.condition = condition
        self.language = language
        self._buffer: list[dict[str, Any]] = []
        self._shard = 0
        self.n_rows = 0

    def write_batch(
        self,
        layer: int,
        seq_ids: list[int],
        seq_len: int,
        fingerprint: str,
        topk_idx: Any,
        topk_weight: Any,
        logits: Any | None,
        logit_stride: int = 16,
    ) -> None:
        """Append one gate's output for one forward pass.

        `topk_idx` / `topk_weight` arrive as (batch*seq, top_k); `logits` as
        (batch*seq, num_experts). Rows map back to (sequence, position) by integer division,
        because the sparse MoE block flattens batch and sequence before routing.
        """
        idx = _to_list2d(topk_idx)
        weights = _to_list2d(topk_weight)
        logit_rows = _to_list2d(logits) if logits is not None else None

        for row, (experts, w) in enumerate(zip(idx, weights, strict=True)):
            seq_pos = row % seq_len
            seq_slot = row // seq_len
            seq_id = seq_ids[seq_slot] if seq_slot < len(seq_ids) else seq_slot
            keep_logits = logit_rows is not None and (seq_pos % logit_stride == 0)
            self._buffer.append(
                {
                    "condition": self.condition,
                    "language": self.language,
                    "layer": layer,
                    "seq_id": int(seq_id),
                    "position": int(seq_pos),
                    "fingerprint": fingerprint,
                    "expert_ids": [int(e) for e in experts],
                    "expert_weights": [float(x) for x in w],
                    "logits": [float(x) for x in logit_rows[row]] if keep_logits else None,
                }
            )
            self.n_rows += 1

        if len(self._buffer) >= ROWS_PER_FLUSH:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        path = self.out_dir / f"part-{self._shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self._buffer.clear()
        self._shard += 1
        return path

    def close(self) -> None:
        self.flush()


def _to_list2d(tensor: Any) -> list[list[Any]]:
    """Convert a torch tensor (or anything list-like) to nested Python lists."""
    if tensor is None:
        return []
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().to("cpu")
        if hasattr(tensor, "float") and str(getattr(tensor, "dtype", "")) in {
            "torch.bfloat16",
        }:
            # numpy has no bfloat16; go through float32.
            tensor = tensor.float()
        tensor = tensor.tolist()
    if tensor and not isinstance(tensor[0], list):
        return [list(tensor)]
    return [list(row) for row in tensor]


def read_trace(
    trace_dir: str | Path,
    condition: str | None = None,
    language: str | None = None,
    layers: list[int] | None = None,
    columns: list[str] | None = None,
):
    """Read a routing trace into a pandas DataFrame, pushing filters into Parquet.

    Reading only the columns and layers a metric needs is what keeps 4M-row traces usable on
    a laptop.
    """
    root = Path(trace_dir)
    if condition is not None:
        root_candidate = root / f"condition={condition}"
        if root_candidate.exists():
            root = root_candidate
    if language is not None:
        lang_candidate = root / f"language={language}"
        if lang_candidate.exists():
            root = lang_candidate

    if not root.exists():
        raise FileNotFoundError(f"no routing trace at {root}")

    filters = []
    if layers:
        filters.append(("layer", "in", layers))

    dataset = pq.ParquetDataset(root, filters=filters or None)
    table = dataset.read(columns=columns)
    df = table.to_pandas()

    # Hive-style partition columns are recovered from the path when the read started below
    # the partition level, so downstream code can always rely on them being present.
    if "condition" not in df.columns and condition is not None:
        df["condition"] = condition
    if "language" not in df.columns and language is not None:
        df["language"] = language
    return df
