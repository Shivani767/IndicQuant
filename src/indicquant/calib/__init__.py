"""Calibration corpus construction.

The invariant that makes the headline comparison valid: every calibration corpus is built to
the SAME token budget. B vs. E must vary distribution only. See `build.py`.
"""

from indicquant.calib.build import (
    CalibrationBudgetError,
    CalibrationCorpus,
    build_corpus,
    verify_budget_parity,
)

__all__ = [
    "CalibrationBudgetError",
    "CalibrationCorpus",
    "build_corpus",
    "verify_budget_parity",
]
