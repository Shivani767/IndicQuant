"""ingest → split → preprocess → infer → validate → assemble."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from indicquant.agent.script import ascii_digits
from indicquant.harness.extract import extract, route_doc_type, validate
from indicquant.harness.schemas import SCHEMAS

Emit = Callable[[dict[str, Any]], None]


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


def _pages_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
    pages = doc.get("pages")
    if pages:
        return [{"text": str(p.get("text") or ""), "boxes": p.get("boxes") or []} for p in pages]
    text = str(doc.get("text") or "")
    chunks = [c.strip() for c in text.split("\f") if c.strip()] or [text]
    return [{"text": c, "boxes": []} for c in chunks]


def _merge(pages_fields: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fields in pages_fields:
        for key, value in fields.items():
            if value not in (None, "") and out.get(key) in (None, ""):
                out[key] = value
    return out


def run_pipeline(doc: dict[str, Any], emit: Emit | None = None) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def emit_stage(row: dict[str, Any]) -> None:
        stages.append(row)
        if emit:
            emit({"type": "stage", **row})

    def ingest() -> dict[str, Any]:
        pages = _pages_of(doc)
        state["raw_pages"] = pages
        return {"pages": len(pages), "chars": sum(len(p["text"]) for p in pages)}

    emit_stage(_stage("ingest", ingest))

    def split() -> dict[str, Any]:
        state["pages"] = list(state["raw_pages"])
        return {"fanout": len(state["pages"])}

    emit_stage(_stage("split", split))

    def preprocess() -> dict[str, Any]:
        cleaned = []
        scripts = []
        for page in state["pages"]:
            text = ascii_digits(page["text"])
            cleaned.append({**page, "text": text, "raw": page["text"]})
            scripts.append(_script_hint(page["text"]))
        state["pages"] = cleaned
        return {"script": scripts, "deskew": "identity"}

    emit_stage(_stage("preprocess", preprocess))

    def infer() -> dict[str, Any]:
        hint = doc.get("doc_type")
        per_page = []
        for page in state["pages"]:
            kind = hint or route_doc_type(page["text"])
            result = extract(page["text"], kind)
            per_page.append(result)
        state["per_page"] = per_page
        kinds = [p["doc_type"] for p in per_page]
        majority = max(set(kinds), key=kinds.count) if kinds else (hint or "pan")
        state["doc_type"] = hint or majority
        return {"doc_type": state["doc_type"], "pages": len(per_page)}

    emit_stage(_stage("infer", infer))

    def validate_stage() -> dict[str, Any]:
        merged = _merge([p["fields"] for p in state["per_page"]])
        report = validate(merged, state["doc_type"])
        if not report["ok"]:
            retried = []
            for page in state["pages"]:
                retried.append(extract(page["text"], state["doc_type"]))
            state["per_page"] = retried
            merged = _merge([p["fields"] for p in retried])
            report = validate(merged, state["doc_type"])
            report["retried"] = True
        state["fields"] = merged
        state["validation"] = report
        return report

    emit_stage(_stage("validate", validate_stage))

    def assemble() -> dict[str, Any]:
        conf: dict[str, float] = {}
        for page in state["per_page"]:
            for key, value in page["confidence"].items():
                conf[key] = max(conf.get(key, 0.0), value)
        schema = SCHEMAS[state["doc_type"]]
        payload = {
            "doc_type": state["doc_type"],
            "title": schema["title"],
            "fields": state["fields"],
            "confidence": conf,
            "validation": state["validation"],
            "pages": len(state["pages"]),
        }
        state["result"] = payload
        return {"fields": len(state["fields"]), "ok": state["validation"]["ok"]}

    emit_stage(_stage("assemble", assemble))

    total_ms = round(sum(s["ms"] for s in stages), 2)
    return {
        "job_id": doc.get("job_id") or uuid.uuid4().hex[:12],
        "ok": bool((state.get("validation") or {}).get("ok")),
        "doc_type": state.get("doc_type"),
        "result": state.get("result"),
        "stages": stages,
        "latency_ms": total_ms,
        "cost": {"unit": "cpu_ms", "total": total_ms, "by_stage": {s["name"]: s["ms"] for s in stages}},
    }


def _script_hint(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "Devanagari"
        if 0x0B80 <= o <= 0x0BFF:
            return "Tamil"
        if 0x0B00 <= o <= 0x0B7F:
            return "Odia"
    return "Latin"
