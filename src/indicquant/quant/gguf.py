"""GGUF / llama.cpp path — Condition D.

The deployment reality check. GGUF is how quantized models actually reach laptops and edge
devices, and Sarvam already ships `sarvamai/sarvam-30b-gguf`.

Worth being precise about what "calibration" means here, because it is NOT the same lever as
AWQ/GPTQ. K-quants use an **importance matrix** (imatrix) built by running text through the
model and accumulating per-channel activation importance. The calibration-language lever is
the imatrix corpus. Same question, genuinely different mechanism — which makes Condition D an
independent test rather than a restatement of Condition B.

Consequence for RQ4: llama.cpp fuses the MoE block, so `SarvamMoEGate` is not a live Python
module and gate hooks bind nothing. Condition D records `capture_routing: false` explicitly so
an absent trace is never read as an absence of drift.
"""

from __future__ import annotations

from pathlib import Path


def build_imatrix(
    gguf_model_path: str | Path,
    calibration_text_path: str | Path,
    output_path: str | Path,
    n_chunks: int = 512,
) -> Path:
    """Build an importance matrix from a calibration corpus.

    This is the calibration-language lever for Condition D. Wraps
    `llama-imatrix -m <model> -f <corpus> -o <imatrix>`.
    """
    raise NotImplementedError(
        "Phase 1 deliverable. Requires a llama.cpp build.\n"
        "Pipeline: convert_hf_to_gguf.py -> llama-imatrix -> llama-quantize.\n"
        "Verify first that convert_hf_to_gguf.py supports the sarvam_moe architecture — "
        "sarvamai/sarvam-30b-gguf exists, so a conversion path does, but it may live in a "
        "fork or need a patch. Check before budgeting time for Condition D."
    )


def quantize_gguf(
    hf_model_path: str | Path,
    output_path: str | Path,
    quant_type: str = "Q4_K_M",
    imatrix_path: str | Path | None = None,
) -> Path:
    """Convert to GGUF and quantize, optionally with an imatrix."""
    raise NotImplementedError(
        "Phase 1 deliverable. Requires a llama.cpp build. See build_imatrix() for the "
        "prerequisite check on sarvam_moe conversion support."
    )
