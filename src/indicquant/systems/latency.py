"""Throughput and latency measurement.

Runs on vLLM, which fuses the MoE block into custom kernels. That makes it the right tool for
systems numbers and the wrong tool for routing — gate hooks bind nothing there. The split is
deliberate and enforced: routing capture happens on the HF path (ARCHITECTURE.md §6.3).

Per-language measurement matters here in a way it does not for dense models: Indic scripts
have higher tokenizer fertility, so an equal-length *prompt* is a longer *sequence*. Reporting
tokens/sec without reporting fertility alongside it hides the real per-language cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LatencyResult:
    condition: str
    language: str
    n_requests: int
    prompt_tokens_mean: float
    ttft_ms_mean: float
    ttft_ms_p95: float
    tokens_per_sec_mean: float
    e2e_ms_p95: float
    peak_memory_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_latency(
    checkpoint_path: str,
    prompts: list[str],
    condition: str,
    language: str,
    max_new_tokens: int = 128,
    **vllm_kwargs: Any,
) -> LatencyResult:
    """Measure TTFT, throughput and P95 latency for one condition/language pair.

    Requires the `serve` extra and a CUDA device.
    """
    raise NotImplementedError(
        "Phase 1 deliverable. Requires the `serve` extra (vllm) in its own venv — it cannot "
        "coexist with `quant` — and a CUDA device. "
        "Note that sarvam-30b ships hotpatch_vllm.py — vLLM support for sarvam_moe is not "
        "automatic and must be verified on the first GPU session."
    )
