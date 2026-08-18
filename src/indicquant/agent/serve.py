"""Product HTTP UI. Open http://127.0.0.1:7860 — same agent as the CLI."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from indicquant import __version__
from indicquant.agent.chat import build_agent
from indicquant.agent.llm import LLM
from indicquant.agent.pending import PendingStore
from indicquant.agent.session import SessionHub, default_sessions_dir
from indicquant.agent.trace import format_stages
from indicquant.harness.jobs import JobBackend, open_jobs, sample_catalog

STATIC = Path(__file__).resolve().parent / "static" / "index.html"

EXAMPLES = [
    "भारत की राजधानी क्या है?",
    "Explain photosynthesis in simple English",
    "Python में list reverse कैसे करते हैं?",
    "मुंबई से दिल्ली के लिए 2 टिकट बुक करो",
    "500 रुपये पर 18 प्रतिशत GST कितना?",
]


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    data = json.loads(raw.decode() or "{}")
    return data if isinstance(data, dict) else {}


def _wants_stream(handler: BaseHTTPRequestHandler) -> bool:
    return "text/event-stream" in (handler.headers.get("Accept") or "")


def _sse_open(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _sse_send(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    handler.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
    handler.wfile.flush()


def _turn_payload(sess, turn) -> dict[str, Any]:
    payload = turn.to_dict()
    payload["notes"] = list(sess.agent.memory.notes)
    payload["records"] = [r.to_dict() for r in sess.agent.memory.records]
    payload["session_id"] = sess.id
    payload["stages"] = format_stages(turn, verbose=True)
    return payload


def make_handler(
    hub: SessionHub, backend: str, jobs: JobBackend | None = None
) -> type[BaseHTTPRequestHandler]:
    jobs = jobs or open_jobs()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path in {"/", "/index.html"}:
                body = STATIC.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/meta":
                _json(
                    self,
                    200,
                    {
                        "backend": backend,
                        "version": __version__,
                        "examples": EXAMPLES,
                        "loop": ["ingest", "split", "preprocess", "infer", "validate", "assemble"],
                        "autoopt_loop": ["ingest", "preprocess", "mer", "compile", "validate", "solve"],
                        "samples": sample_catalog(),
                    },
                )
                return
            if path in {"/api/health", "/health"}:
                payload = jobs.health()
                payload["version"] = __version__
                payload["backend"] = backend
                _json(self, 200 if payload.get("ok") else 503, payload)
                return
            if path == "/api/eval":
                from indicquant.eval.docs import run_eval as run_doc_eval

                _json(self, 200, run_doc_eval())
                return
            if path == "/api/eval/autoopt":
                from indicquant.eval.autoopt import run_eval as run_autoopt_eval

                _json(self, 200, run_autoopt_eval())
                return
            if path == "/api/eval/agent":
                from indicquant.eval.harness import run_eval

                _json(self, 200, run_eval())
                return
            if path == "/api/docs":
                _json(self, 200, {"samples": sample_catalog()})
                return
            if path == "/api/jobs":
                _json(self, 200, {"jobs": jobs.list()})
                return
            if path.startswith("/api/jobs/"):
                jid = path.split("/")[3]
                try:
                    _json(self, 200, jobs.get(jid))
                except KeyError:
                    _json(self, 404, {"error": "unknown job"})
                return
            if path == "/api/pending":
                rows = [
                    {
                        "id": a.id,
                        "status": a.status,
                        "preview": a.preview,
                        "expires_at": a.expires_at,
                    }
                    for a in PendingStore().list(include_closed=True)[:20]
                ]
                _json(self, 200, {"pending": rows})
                return
            if path == "/api/sessions":
                _json(self, 200, {"sessions": hub.list()})
                return
            if path.startswith("/api/sessions/"):
                sid = path.split("/")[3]
                try:
                    sess = hub.get(sid)
                except KeyError:
                    _json(self, 404, {"error": "unknown session"})
                    return
                _json(
                    self,
                    200,
                    {
                        "id": sess.id,
                        "title": sess.title,
                        "created_at": sess.created_at,
                        "notes": list(sess.agent.memory.notes),
                        "records": [r.to_dict() for r in sess.agent.memory.records],
                        "turns": sess.log.turns() if sess.log else [],
                        "pending": sess.agent._pending is not None,
                        "pending_id": (sess.agent._pending or {}).get("action_id"),
                    },
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path == "/api/jobs":
                    body = _read_json(self)
                    if body.get("sample_id"):
                        try:
                            record = jobs.submit_sample(str(body["sample_id"]))
                        except KeyError as exc:
                            _json(self, 404, {"error": f"unknown sample: {exc}"})
                            return
                    elif body.get("image_b64") or body.get("pipeline") == "autoopt":
                        record = jobs.submit(
                            {
                                "text": str(body.get("text") or ""),
                                "pipeline": "autoopt",
                                "image_b64": body.get("image_b64"),
                                "doc_type": "autoopt",
                            }
                        )
                    else:
                        text = str(body.get("text") or "").strip()
                        if not text:
                            raise ValueError("provide sample_id or text")
                        record = jobs.submit_text(
                            text,
                            str(body["doc_type"]) if body.get("doc_type") else None,
                            str(body["pipeline"]) if body.get("pipeline") else None,
                        )
                    _json(self, 200, record)
                    return
                if path == "/api/sessions":
                    sess = hub.create()
                    _json(self, 200, {"id": sess.id, "created_at": sess.created_at})
                    return
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[0] == "api" and parts[1] == "sessions":
                    sid = parts[2]
                    action = parts[3] if len(parts) > 3 else ""
                    sess = hub.get(sid)
                    if action == "chat":
                        message = str(_read_json(self).get("message") or "").strip()
                        if not message:
                            raise ValueError("empty message")
                        if _wants_stream(self):
                            _stream_turn(self, lambda emit: sess.agent.act(message, emit=emit), sess, hub, message)
                            return
                        turn = sess.agent.act(message)
                        hub.record(sess, message, turn)
                        _json(self, 200, _turn_payload(sess, turn))
                        return
                    if action == "confirm":
                        if _wants_stream(self):
                            _stream_turn(self, lambda emit: sess.agent.confirm(emit=emit), sess, hub, "[confirm]")
                            return
                        turn = sess.agent.confirm()
                        hub.record(sess, "[confirm]", turn)
                        _json(self, 200, _turn_payload(sess, turn))
                        return
                    if action == "cancel":
                        turn = sess.agent.cancel()
                        hub.record(sess, "[cancel]", turn)
                        _json(self, 200, _turn_payload(sess, turn))
                        return
                    if action == "reset":
                        sess.agent.reset()
                        _json(self, 200, {"ok": True})
                        return
                self.send_error(404)
            except KeyError:
                _json(self, 404, {"error": "unknown session"})
            except Exception as exc:  # noqa: BLE001 — surface to the UI
                _json(self, 400, {"error": str(exc)})

    return Handler


def _stream_turn(handler, run, sess, hub, user_text: str) -> None:
    _sse_open(handler)

    def emit(event: dict[str, Any]) -> None:
        _sse_send(handler, event)

    try:
        turn = run(emit)
        hub.record(sess, user_text, turn)
        _sse_send(handler, {"type": "done", "turn": _turn_payload(sess, turn)})
    except Exception as exc:  # noqa: BLE001 — surface to the UI
        _sse_send(handler, {"type": "error", "error": str(exc)})


def serve(host: str = "127.0.0.1", port: int = 7860, llm: LLM | None = None) -> None:
    def make_agent():
        if llm is None:
            raise RuntimeError("chat disabled: no LLM in this process")
        return build_agent(llm, auto_confirm=False, persist=False)

    hub = SessionHub(make_agent, default_sessions_dir())
    backend = "docs-only"
    if llm is not None:
        backend = getattr(llm, "model", None) or llm.name
    httpd = ThreadingHTTPServer((host, port), make_handler(hub, backend, open_jobs()))
    print(f"IndicQuant UI → http://{host}:{port}  (backend: {backend})")
    httpd.serve_forever()
