"""HTTP product API — same agent as the CLI."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from indicquant.agent.env import AgentEnv
from indicquant.agent.llm import LLMOutput, ScriptedLLM
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent
from indicquant.agent.serve import make_handler
from indicquant.agent.session import SessionHub
from indicquant.agent.tools import user_tools
from indicquant.agent.types import ToolCall
from indicquant.harness.jobs import JobStore


def _server(tmp_path, script: list[LLMOutput]):
    def make() -> Agent:
        store = EpisodicStore()
        return Agent(
            ScriptedLLM(list(script)),
            AgentEnv(tools=user_tools(store)),
            memory=store,
            auto_confirm=False,
        )

    hub = SessionHub(make, tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(hub, "scripted", JobStore(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _req(httpd, method: str, path: str, body: dict | None = None) -> dict:
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    conn.request(method, path, body=payload, headers=headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode())
    conn.close()
    return data


def test_meta_and_session_chat(tmp_path) -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[
                ToolCall("book_ticket", {"origin": "मुंबई", "destination": "दिल्ली", "passengers": 2})
            ],
        ),
        LLMOutput(content="टिकट बुक हो गई।"),
    ]
    httpd = _server(tmp_path, script)
    try:
        meta = _req(httpd, "GET", "/api/meta")
        assert "ingest" in meta["loop"]
        sess = _req(httpd, "POST", "/api/sessions")
        sid = sess["id"]
        turn = _req(
            httpd,
            "POST",
            f"/api/sessions/{sid}/chat",
            {"message": "मुंबई से दिल्ली के लिए 2 टिकट बुक करो"},
        )
        assert turn["pending_id"]
        assert turn["pending"]["preview"]
        done = _req(httpd, "POST", f"/api/sessions/{sid}/confirm")
        tools = [e for e in done["events"] if e["kind"] == "tool"]
        assert tools
        assert "BOMDEL" in tools[0]["data"]["result"]["output"]
        assert "टिकट" in done["reply"]
    finally:
        httpd.shutdown()


def test_sse_chat_streams_pending(tmp_path) -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[
                ToolCall("book_ticket", {"origin": "मुंबई", "destination": "दिल्ली", "passengers": 2})
            ],
        ),
        LLMOutput(content="टिकट बुक हो गई। PNR BOMDEL02"),
    ]
    httpd = _server(tmp_path, script)
    try:
        sess = _req(httpd, "POST", "/api/sessions")
        conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        payload = json.dumps({"message": "मुंबई से दिल्ली के लिए 2 टिकट बुक करो"}).encode()
        conn.request(
            "POST",
            f"/api/sessions/{sess['id']}/chat",
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        res = conn.getresponse()
        assert "event-stream" in (res.getheader("Content-Type") or "")
        raw = res.read().decode()
        conn.close()
        assert '"type": "pending"' in raw or '"type":"pending"' in raw
        assert "uv run" not in raw
        assert "मुंबई" in raw
    finally:
        httpd.shutdown()


def test_document_job_api(tmp_path) -> None:
    httpd = _server(tmp_path, [])
    try:
        catalog = _req(httpd, "GET", "/api/docs")
        assert any(s["id"] == "pan_hi" for s in catalog["samples"])
        job = _req(httpd, "POST", "/api/jobs", {"sample_id": "pan_hi"})
        assert job["ok"] is True
        assert job["result"]["fields"]["pan"] == "ABCDE1234F"
        names = [s["name"] for s in job["stages"]]
        assert names == ["ingest", "split", "preprocess", "infer", "validate", "assemble"]
        loaded = _req(httpd, "GET", f"/api/jobs/{job['id']}")
        assert loaded["id"] == job["id"]
        opt = _req(httpd, "POST", "/api/jobs", {"sample_id": "opt_lp2"})
        assert opt["ok"] is True
        assert opt["result"]["solution"]["objective"] == 18.0
        assert [s["name"] for s in opt["stages"]][2] == "mer"
        health = _req(httpd, "GET", "/api/health")
        assert health["ok"] is True
        assert health["store"] == "files"
        missing = _req(httpd, "POST", "/api/jobs", {"sample_id": "does_not_exist"})
        assert "unknown sample" in missing["error"]
        assert "session" not in missing["error"]
    finally:
        httpd.shutdown()
