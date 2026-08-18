"""AutoOpt golden set: numeric match, or a required failure at a named stage."""

from __future__ import annotations

from typing import Any

from indicquant.autoopt.pipeline import run_autoopt
from indicquant.autoopt.samples import load_cases


def _close(got: Any, want: Any, tol: float = 0.05) -> bool:
    try:
        return abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        return False


def score_case(case: dict[str, Any]) -> dict[str, Any]:
    out = run_autoopt(case)
    sol = (out.get("result") or {}).get("solution") or {}
    expected = case.get("expected") or {}
    errors: list[str] = []
    hits = 0
    n = 0

    if case.get("expect_ok") is False:
        n = 1
        if out.get("ok"):
            errors.append("expected pipeline failure")
        else:
            hits = 1
        stage = case.get("fail_stage")
        if stage:
            n += 1
            hit = any(s["name"] == stage and not s["ok"] for s in out.get("stages") or [])
            if hit:
                hits += 1
            else:
                errors.append(f"expected {stage} to fail")
    else:
        want_obj = expected.get("objective")
        if want_obj is not None:
            n += 1
            got = sol.get("objective")
            if _close(got, want_obj):
                hits += 1
            else:
                errors.append(f"objective: got {got!r} want {want_obj!r}")
        for name, want in (expected.get("x") or {}).items():
            n += 1
            got = (sol.get("x") or {}).get(name)
            if _close(got, want):
                hits += 1
            else:
                errors.append(f"{name}: got {got!r} want {want!r}")
        if not out.get("ok"):
            errors.append("pipeline failed")

    n = n or 1
    return {
        "id": case["id"],
        "ok": not errors,
        "doc_type": out.get("doc_type"),
        "field_accuracy": round(hits / n, 3),
        "latency_ms": out.get("latency_ms"),
        "method": sol.get("method"),
        "errors": errors,
        "stages": [s["name"] for s in out.get("stages") or []],
    }


def run_eval() -> dict[str, Any]:
    scores = [score_case(case) for case in load_cases()]
    n = len(scores) or 1
    acc = [s["field_accuracy"] for s in scores]
    return {
        "suite": "autoopt",
        "cases": len(scores),
        "task_completion_rate": round(sum(1 for s in scores if s["ok"]) / n, 3),
        "field_accuracy_mean": round(sum(acc) / n, 3),
        "latency_ms_mean": round(sum(float(s["latency_ms"] or 0) for s in scores) / n, 2),
        "scores": scores,
    }
