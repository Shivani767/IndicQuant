"""Reproducible eval harness."""

from __future__ import annotations

from indicquant.eval.harness import run_eval


def test_eval_suite_passes() -> None:
    report = run_eval()
    assert report["cases"] >= 6
    assert report["task_completion_rate"] == 1.0
    assert report["hallucination_catch_rate"] == 1.0
    failed = [s for s in report["scores"] if not s["ok"]]
    assert failed == []
