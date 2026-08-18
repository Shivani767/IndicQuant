"""Labelled-scan eval: noisy OCR → small MER → compile → solve.

PNGs under evals/scans/ are the pages. pytest does not require Tesseract;
it runs the grammar MER on stored ocr_noise. Tesseract is an extra row when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indicquant.autoopt.m2 import compile_latex_retry
from indicquant.autoopt.m3 import Infeasible, solve_program
from indicquant.autoopt.mer import recognise

DEFAULT = Path(__file__).resolve().parents[3] / "evals" / "scans.json"


def scans_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "evals" / "scans"
        if candidate.is_dir():
            return candidate
    return DEFAULT.parent / "scans"


def load_scans(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or DEFAULT
    if not target.exists():
        bundled = Path(__file__).resolve().parents[1] / "eval" / "scans.json"
        target = bundled if bundled.exists() else target
    return json.loads(target.read_text(encoding="utf-8"))


def _close(got: Any, want: Any, tol: float = 0.05) -> bool:
    try:
        return abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        return False


def score_scan(case: dict[str, Any]) -> dict[str, Any]:
    mer = recognise(str(case["ocr_noise"]))
    errors: list[str] = []
    try:
        program, _retried = compile_latex_retry(mer["latex"])
        sol = solve_program(program)
    except (ValueError, Infeasible) as exc:
        return {
            "id": case["id"],
            "ok": False,
            "mer": mer["latex"],
            "repairs": mer.get("repairs") or [],
            "errors": [str(exc)],
            "field_accuracy": 0.0,
            "latency_ms": 0.0,
        }
    expected = case.get("expected") or {}
    hits, n = 0, 0
    if expected.get("objective") is not None:
        n += 1
        if _close(sol.get("objective"), expected["objective"]):
            hits += 1
        else:
            errors.append(f"objective {sol.get('objective')!r} != {expected['objective']!r}")
    for name, want in (expected.get("x") or {}).items():
        n += 1
        got = (sol.get("x") or {}).get(name)
        if _close(got, want):
            hits += 1
        else:
            errors.append(f"{name}: {got!r} != {want!r}")
    n = n or 1
    png = scans_dir() / str(case.get("image") or "")
    return {
        "id": case["id"],
        "ok": not errors,
        "mer": mer["latex"],
        "repairs": mer.get("repairs") or [],
        "method": sol.get("method"),
        "field_accuracy": round(hits / n, 3),
        "errors": errors,
        "has_png": png.is_file(),
        "latency_ms": 0.0,
    }


def run_eval() -> dict[str, Any]:
    scores = [score_scan(case) for case in load_scans()]
    n = len(scores) or 1
    acc = [s["field_accuracy"] for s in scores]
    return {
        "suite": "scans",
        "cases": len(scores),
        "task_completion_rate": round(sum(1 for s in scores if s["ok"]) / n, 3),
        "field_accuracy_mean": round(sum(acc) / n, 3),
        "pngs": sum(1 for s in scores if s.get("has_png")),
        "scores": scores,
    }
