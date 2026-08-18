"""Quantization backends.

`llm-compressor` is the primary path (AutoAWQ was archived in May 2025 — see
ARCHITECTURE.md §1.1). `GPTQModel` provides an independent second implementation so a
finding cannot be an artifact of one library. `gguf` covers the deployment path people
actually run.
"""

from indicquant.quant.sarvam_moe_adapter import (
    AWQ_MAPPINGS,
    IgnorePolicy,
    expert_parameter_share,
    preflight_checks,
    tokens_per_expert_fraction,
)

__all__ = [
    "AWQ_MAPPINGS",
    "IgnorePolicy",
    "expert_parameter_share",
    "preflight_checks",
    "tokens_per_expert_fraction",
]
