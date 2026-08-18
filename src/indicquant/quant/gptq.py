"""GPTQModel path — the independent second implementation.

Condition C runs through llm-compressor's GPTQModifier by default. This module exists so the
same condition can be reproduced under a completely different codebase (`GPTQModel` v7.3.2,
actively maintained).

Why bother: if the calibration-language gap appears under llm-compressor and vanishes under
GPTQModel, the finding is about one library's implementation, not about calibration language.
That is exactly the objection a reviewer raises, and it is cheap to pre-empt on a subset of
languages rather than the full sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def quantize_with_gptqmodel(
    hf_id: str,
    calibration_texts: list[str],
    output_dir: str | Path,
    bits: int = 4,
    group_size: int = 128,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """Quantize with GPTQModel as a cross-check on the llm-compressor result."""
    raise NotImplementedError(
        "Phase 1 cross-check. Requires the `quant` extra (gptqmodel) and a CUDA device.\n"
        "Run on the Phase 0 language subset only — this exists to rule out a single-library "
        "artifact, not to produce headline numbers.\n"
        "Note: GPTQModel needs its own module map for sarvam's fused "
        "attention.query_key_value naming; reuse the patterns in sarvam_moe_adapter.py."
    )


def compare_backends(
    llmcompressor_results: dict[str, Any],
    gptqmodel_results: dict[str, Any],
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Check that two backends agree on the direction and rough size of the effect.

    Agreement on *direction* is what matters. Exact accuracy parity is not expected — the
    implementations differ in reordering, damping, and grouping details.
    """
    shared = set(llmcompressor_results) & set(gptqmodel_results)
    rows = []
    for key in sorted(shared):
        a, b = llmcompressor_results[key], gptqmodel_results[key]
        rows.append(
            {
                "key": key,
                "llmcompressor": a,
                "gptqmodel": b,
                "abs_diff": abs(a - b),
                "same_sign": (a >= 0) == (b >= 0),
                "within_tolerance": abs(a - b) <= tolerance,
            }
        )
    return {
        "rows": rows,
        "all_same_sign": all(r["same_sign"] for r in rows) if rows else False,
        "n_compared": len(rows),
    }
