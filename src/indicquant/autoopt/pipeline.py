"""ingest → preprocess → mer → compile → validate → solve."""

from __future__ import annotations

import base64
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from indicquant.autoopt.m1 import mer_from_text, ocr_png
from indicquant.autoopt.m2 import compile_latex_retry
from indicquant.autoopt.m3 import Infeasible, solve_program
from indicquant.autoopt.mer import recognise

Emit = Callable[[dict[str, Any]], None]
_OPT_HINT = re.compile(
    r"(?:\\?min(?:imize)?|\\?max(?:imize)?).{0,200}(?:s\.t\.|subject\s+to|\\le|\\ge|<=|>=|≤|≥)",
    re.I | re.S,
)


def looks_like_opt(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 8:
        return False
    return bool(_OPT_HINT.search(body))


def _stage(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        detail = fn()
        ok = True
        error = None
    except Exception as exc:  # noqa: BLE001 — surface on the job record
        detail = {}
        ok = False
        error = str(exc)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    row: dict[str, Any] = {"name": name, "ok": ok, "ms": ms, "detail": detail}
    if error:
        row["error"] = error
    return row


def _input_text(doc: dict[str, Any]) -> str:
    if doc.get("text"):
        return str(doc["text"])
    pages = doc.get("pages") or []
    return "\n".join(str(p.get("text") or "") for p in pages)


def _image_bytes(doc: dict[str, Any]) -> bytes | None:
    raw = doc.get("image")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    b64 = doc.get("image_b64")
    if b64:
        return base64.b64decode(str(b64))
    return None


def run_autoopt(doc: dict[str, Any], emit: Emit | None = None) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def emit_stage(row: dict[str, Any]) -> None:
        stages.append(row)
        if emit:
            emit({"type": "stage", **row})

    def ingest() -> dict[str, Any]:
        image = _image_bytes(doc)
        text = _input_text(doc)
        state["image"] = image
        state["text"] = text
        return {
            "pages": 1,
            "chars": len(text),
            "has_image": image is not None,
            "source": "image" if image else "text",
        }

    emit_stage(_stage("ingest", ingest))

    def preprocess() -> dict[str, Any]:
        if state.get("image"):
            from indicquant.autoopt.m1 import preprocess_image

            state["image"] = preprocess_image(state["image"])
            return {"deskew": "pad-contrast-unsharp", "canvas": "768x1024"}
        return {"deskew": "identity", "note": "text path skips image preprocess"}

    emit_stage(_stage("preprocess", preprocess))

    def mer() -> dict[str, Any]:
        if state.get("image") and not state.get("text"):
            ocr = ocr_png(state["image"])
            out = recognise(ocr["latex"])
            out["source"] = "tesseract+mer"
        else:
            base = mer_from_text(state["text"])
            out = recognise(base["latex"])
            out["source"] = "text+mer" if out.get("repairs") else "text"
        state["mer"] = out
        return {
            "latex": out["latex"],
            "source": out["source"],
            "chars": out["chars"],
            "repairs": out.get("repairs") or [],
        }

    emit_stage(_stage("mer", mer))

    def compile_stage() -> dict[str, Any]:
        latex = (state.get("mer") or {}).get("latex") or ""
        program, retried = compile_latex_retry(latex)
        state["program"] = program
        state["compile_retried"] = retried
        return {
            "sense": program.sense,
            "variables": program.variables,
            "constraints": len(program.constraints),
            "quadratic": list(program.quadratic),
            "retried": retried,
        }

    emit_stage(_stage("compile", compile_stage))

    def validate_stage() -> dict[str, Any]:
        program = state.get("program")
        if program is None:
            raise ValueError("M2 checkpoint failed — no compiled program")
        if not program.variables or not program.constraints:
            raise ValueError("M2 checkpoint: incomplete model")
        report = {
            "ok": True,
            "retried": bool(state.get("compile_retried")),
            "checkpoint": "latex → structured program",
        }
        state["validation"] = report
        return report

    emit_stage(_stage("validate", validate_stage))

    def solve_stage() -> dict[str, Any]:
        program = state["program"]
        try:
            sol = solve_program(program)
        except Infeasible as exc:
            raise ValueError(f"M3 infeasible: {exc}") from exc
        state["solution"] = sol
        return {
            "status": sol["status"],
            "method": sol["method"],
            "objective": sol["objective"],
            "x": sol["x"],
        }

    emit_stage(_stage("solve", solve_stage))

    ok = all(s["ok"] for s in stages) and state.get("solution") is not None
    program = state.get("program")
    mer_out = state.get("mer") or {}
    payload = {
        "doc_type": "autoopt",
        "title": "AutoOpt (arXiv:2510.21436) harness",
        "latex": mer_out.get("latex"),
        "program": program.to_dict() if program is not None else None,
        "solution": state.get("solution"),
        "validation": state.get("validation"),
        "pages": 1,
    }
    total_ms = round(sum(s["ms"] for s in stages), 2)
    return {
        "job_id": doc.get("job_id") or uuid.uuid4().hex[:12],
        "ok": ok,
        "doc_type": "autoopt",
        "result": payload,
        "stages": stages,
        "latency_ms": total_ms,
        "cost": {"unit": "cpu_ms", "total": total_ms, "by_stage": {s["name"]: s["ms"] for s in stages}},
    }
