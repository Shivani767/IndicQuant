"""Small MER on labelled-scan OCR noise."""

from __future__ import annotations

from indicquant.autoopt.mer import recognise
from indicquant.eval.scans import run_eval


def test_repair_sense_and_le() -> None:
    out = recognise("maxlmize 3*x + 4*y\nsubject to\nx + 2*y ≤ 8\nx >= 0\ny >= 0")
    assert out["latex"].lower().startswith("maximize")
    assert "<=" in out["latex"]
    assert "≤" not in out["latex"]


def test_scan_golden_set() -> None:
    report = run_eval()
    assert report["cases"] >= 4
    failed = [s for s in report["scores"] if not s["ok"]]
    assert failed == [], failed
    assert report["task_completion_rate"] == 1.0
    assert report["pngs"] >= 4
