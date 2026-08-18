"""Golden-set field accuracy. No GPU — extractors are deterministic."""

from __future__ import annotations

from typing import Any

from indicquant.harness.pipeline import run_pipeline
from indicquant.harness.samples import load_cases


def _eq(got: Any, want: Any) -> bool:
    if isinstance(want, float) or isinstance(got, float):
        try:
            return abs(float(got) - float(want)) < 1e-6
        except (TypeError, ValueError):
            return False
    return got == want


def score_case(case: dict[str, Any]) -> dict[str, Any]:
    out = run_pipeline(case)
    fields = (out.get("result") or {}).get("fields") or {}
    expected = case.get("expected") or {}
    errors: list[str] = []
    hits = 0
    for key, want in expected.items():
        got = fields.get(key)
        if _eq(got, want):
            hits += 1
        else:
            errors.append(f"{key}: got {got!r} want {want!r}")
    if case.get("doc_type") and out.get("doc_type") != case["doc_type"]:
        errors.append(f"routed {out.get('doc_type')} != {case['doc_type']}")
    n = len(expected) or 1
    return {
        "id": case["id"],
        "ok": not errors,
        "doc_type": out.get("doc_type"),
        "field_accuracy": round(hits / n, 3),
        "latency_ms": out.get("latency_ms"),
        "errors": errors,
        "stages": [s["name"] for s in out.get("stages") or []],
    }


def run_eval() -> dict[str, Any]:
    scores = [score_case(case) for case in load_cases()]
    n = len(scores) or 1
    acc = [s["field_accuracy"] for s in scores]
    return {
        "suite": "documents",
        "cases": len(scores),
        "task_completion_rate": round(sum(1 for s in scores if s["ok"]) / n, 3),
        "field_accuracy_mean": round(sum(acc) / n, 3),
        "latency_ms_mean": round(sum(float(s["latency_ms"] or 0) for s in scores) / n, 2),
        "scores": scores,
    }
