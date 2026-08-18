"""Router capture via forward hooks on `SarvamMoEGate`.

Why this is simple (ARCHITECTURE.md §1.3): sarvam's gate returns everything we need from one
place::

    # modeling_sarvam_moe.py:236-245
    class SarvamMoEGate(nn.Module):
        def forward(self, hidden_states):
            ...
            return topk_idx, topk_weight, logits

So a plain `register_forward_hook` gets chosen experts, their weights, and the full 128-dim
router logits without touching the model definition.

The load-bearing constraint: this works only while the gate is a live Python module. It holds
under HF transformers + compressed-tensors. It does NOT hold under vLLM or llama.cpp, which
fuse the MoE block — there, hooks would bind nothing and silently record an empty trace that
looks like a null result. `RouterRecorder` therefore raises if it binds zero hooks rather than
producing an empty trace.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass

from indicquant.routing.storage import RoutingTraceWriter


class RoutingCaptureError(RuntimeError):
    """Raised when router capture cannot be performed as configured.

    Most importantly: raised when zero gate modules are found, which happens when the model
    was loaded through a backend that fuses the MoE block. Failing loudly here prevents an
    empty trace from being mistaken for "no routing drift".
    """


def find_gate_modules(model: Any, gate_suffix: str = "mlp.gate") -> dict[int, Any]:
    """Locate the router gate of every MoE layer, keyed by layer index.

    Returns `{layer_index: module}`. Layers below `first_k_dense_replace` are dense and have
    no gate, so for sarvam-30b this yields layers 1..18 (18 entries), not 0..18.
    """
    gates: dict[int, Any] = {}
    for name, module in model.named_modules():
        if not name.endswith(gate_suffix):
            continue
        layer_idx = _layer_index_from_name(name)
        if layer_idx is not None:
            gates[layer_idx] = module
    return dict(sorted(gates.items()))


def _layer_index_from_name(name: str) -> int | None:
    """Extract the layer index from a dotted module path like `model.layers.7.mlp.gate`."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def sequence_fingerprint(input_ids: Any) -> str:
    """Stable hash of a token sequence.

    This is what enforces the teacher-forcing guard (ARCHITECTURE.md §6.1). Routing traces
    from two conditions are only comparable if they were produced on identical token
    sequences; the fingerprint is recorded per sequence so `topk_agreement` can verify it
    rather than trust it.
    """
    try:
        flat = input_ids.detach().cpu().reshape(-1).tolist()
    except AttributeError:
        flat = list(input_ids)
    payload = ",".join(str(int(t)) for t in flat).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class RouterRecorder:
    """Context manager that records routing decisions for every MoE layer.

    Usage::

        with RouterRecorder(model, out_dir="results/routing", condition="B",
                            language="ta") as rec:
            for batch in batches:
                rec.set_sequence(batch["input_ids"], seq_ids=batch["seq_ids"])
                model(input_ids=batch["input_ids"])
        # trace flushed on exit

    `set_sequence` must be called before each forward pass. The recorder needs the token IDs
    to compute the per-sequence fingerprint, and it has no other way to see them: the hook
    fires on the gate module, which receives hidden states, not tokens.
    """

    model: Any
    out_dir: str | Path
    condition: str
    language: str
    gate_suffix: str = "mlp.gate"
    logit_stride: int = 16
    """Store full 128-dim logits every Nth position. Top-k IDs/weights are always stored in
    full. Storing logits for every token turns a ~5M-row table into a ~600M-row one, and the
    entropy statistics we need are stable under position subsampling."""

    store_logits: bool = True
    seed: int | None = None

    _writer: RoutingTraceWriter | None = field(default=None, init=False, repr=False)
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)
    _gates: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _seq_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _fingerprint: str | None = field(default=None, init=False, repr=False)
    _position_offset: int = field(default=0, init=False, repr=False)
    _seq_len: int = field(default=0, init=False, repr=False)
    _batch_size: int = field(default=0, init=False, repr=False)

    # -- lifecycle ----------------------------------------------------------------------

    def __enter__(self) -> RouterRecorder:
        self._gates = find_gate_modules(self.model, self.gate_suffix)
        if not self._gates:
            raise RoutingCaptureError(
                f"no gate modules matching {self.gate_suffix!r} found. Either the model is "
                "dense, or it was loaded through a backend that fuses the MoE block (vLLM, "
                "llama.cpp). Routing analysis must run on the HF transformers path — see "
                "ARCHITECTURE.md §6.3."
            )
        self._writer = RoutingTraceWriter(
            out_dir=Path(self.out_dir), condition=self.condition, language=self.language
        )
        for layer_idx, module in self._gates.items():
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._writer is not None:
            self._writer.close()

    # -- recording ----------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self._gates)

    @property
    def layer_indices(self) -> list[int]:
        return list(self._gates)

    def set_sequence(self, input_ids: Any, seq_ids: list[int] | None = None) -> None:
        """Register the tokens about to be pushed through the model.

        Must be called before every forward pass. Records the fingerprint that makes
        cross-condition routing comparison verifiable.
        """
        shape = tuple(input_ids.shape)
        if len(shape) == 1:
            self._batch_size, self._seq_len = 1, shape[0]
        else:
            self._batch_size, self._seq_len = shape[0], shape[-1]
        self._seq_ids = list(seq_ids) if seq_ids is not None else list(range(self._batch_size))
        if len(self._seq_ids) != self._batch_size:
            raise RoutingCaptureError(
                f"seq_ids has {len(self._seq_ids)} entries but batch size is {self._batch_size}"
            )
        self._fingerprint = sequence_fingerprint(input_ids)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):  # noqa: ARG001 - torch hook signature
            self._record(layer_idx, output)

        return hook

    def _record(self, layer_idx: int, output: Any) -> None:
        if self._fingerprint is None:
            raise RoutingCaptureError(
                "set_sequence() must be called before each forward pass — the recorder "
                "cannot see token IDs from the gate hook alone."
            )
        if not (isinstance(output, tuple) and len(output) == 3):
            raise RoutingCaptureError(
                "expected SarvamMoEGate to return (topk_idx, topk_weight, logits); got "
                f"{type(output).__name__} of length {len(output) if hasattr(output, '__len__') else 'n/a'}. "
                "The remote modeling code may have changed — re-verify against "
                "modeling_sarvam_moe.py."
            )
        topk_idx, topk_weight, logits = output

        # The gate operates on flattened (batch*seq, hidden), so rows map back to
        # (sequence, position) by integer division.
        assert self._writer is not None
        self._writer.write_batch(
            layer=layer_idx,
            seq_ids=self._seq_ids,
            seq_len=self._seq_len,
            fingerprint=self._fingerprint,
            topk_idx=topk_idx,
            topk_weight=topk_weight,
            logits=logits if self.store_logits else None,
            logit_stride=self.logit_stride,
        )

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    @property
    def n_rows(self) -> int:
        return self._writer.n_rows if self._writer is not None else 0
