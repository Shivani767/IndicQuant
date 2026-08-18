"""LLM backends. The agent core never plans by itself — a model does.

OpenAI-compatible Chat Completions with native `tools` is the protocol Azure, GitHub
Models, Ollama, and OpenAI all speak. A scripted backend exists only so the agent *loop*
can be tested without a network.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from indicquant.agent.types import ToolCall

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMOutput:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMOutput: ...

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMOutput: ...


class ScriptedLLM:
    """Deterministic stand-in for tests. Each call pops the next scripted output."""

    name = "scripted"

    def __init__(self, script: list[LLMOutput]) -> None:
        self._script = list(script)
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMOutput:
        with self._lock:
            if not self._script:
                return LLMOutput(content="(script exhausted)")
            return self._script.pop(0)

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMOutput:
        out = self.complete(messages, tools)
        if on_delta and out.content and not out.tool_calls:
            on_delta(out.content)
        return out


class OpenAICompatLLM:
    """OpenAI, Azure OpenAI, GitHub Models, Ollama `/v1/chat/completions`."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        local_only: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.local_only = local_only

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMOutput:
        if self.local_only:
            host = urlparse(self.base_url).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError(f"offline mode blocked non-local LLM URL {self.base_url}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key and "api-key" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self._completions_url(),
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:800]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM unreachable at {self.base_url}: {exc.reason}") from exc
        return parse_completion(body)

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMOutput:
        if self.local_only:
            host = urlparse(self.base_url).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError(f"offline mode blocked non-local LLM URL {self.base_url}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key and "api-key" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self._completions_url(),
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return _consume_stream(resp, on_delta)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:800]
            if exc.code in {400, 404, 415, 422}:
                out = self.complete(messages, tools)
                if on_delta and out.content and not out.tool_calls:
                    on_delta(out.content)
                return out
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM unreachable at {self.base_url}: {exc.reason}") from exc

    def _completions_url(self) -> str:
        url = f"{self.base_url}/chat/completions"
        if "openai/deployments" in self.base_url and "api-version=" not in url:
            url += "?api-version=2024-10-21"
        return url


def parse_tool_call(text: str) -> ToolCall:
    match = _JSON_OBJECT.search(text)
    if not match:
        return ToolCall("finish", {"answer": text.strip()}, thought="unparseable")
    try:
        obj: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ToolCall("finish", {"answer": text.strip()}, thought="bad json")
    name = str(obj.get("tool") or obj.get("name") or "finish")
    args = obj.get("args") or obj.get("parameters") or {}
    if not isinstance(args, dict):
        args = {}
    thought = str(obj.get("thought", ""))
    return ToolCall(name, args, thought=thought)


def parse_completion(body: dict[str, Any]) -> LLMOutput:
    msg = body["choices"][0]["message"]
    raw = msg.get("tool_calls") or []
    calls: list[ToolCall] = []
    for item in raw:
        fn = item.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(str(fn.get("name") or "finish"), args))
    content = (msg.get("content") or "").strip()
    if not calls and content:
        parsed = parse_tool_call(content)
        if parsed.name != "finish":
            calls.append(parsed)
            content = parsed.thought
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    return LLMOutput(content=content, tool_calls=calls, raw_tool_calls=list(raw), usage=usage)


def _consume_stream(resp: Any, on_delta: Callable[[str], None] | None) -> LLMOutput:
    content_parts: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    mode: str | None = None
    for obj in _iter_stream_json(resp):
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            continue
        piece = delta.get("content")
        if piece:
            content_parts.append(str(piece))
            if mode != "tools":
                mode = "text"
                if on_delta:
                    on_delta(str(piece))
        for item in delta.get("tool_calls") or []:
            if not isinstance(item, dict):
                continue
            mode = "tools"
            idx = int(item.get("index") or 0)
            slot = tools.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if item.get("id"):
                slot["id"] = str(item["id"])
            fn = item.get("function") or {}
            if isinstance(fn, dict):
                if fn.get("name"):
                    slot["name"] += str(fn["name"])
                if fn.get("arguments"):
                    slot["arguments"] += str(fn["arguments"])
    raw: list[dict[str, Any]] = []
    calls: list[ToolCall] = []
    for idx in sorted(tools):
        slot = tools[idx]
        try:
            args = json.loads(slot["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        cid = slot["id"] or f"call_{idx}"
        raw.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
        )
        calls.append(ToolCall(slot["name"] or "finish", args))
    content = "".join(content_parts).strip()
    if not calls and content:
        parsed = parse_tool_call(content)
        if parsed.name != "finish":
            calls.append(parsed)
            content = parsed.thought
    return LLMOutput(content=content, tool_calls=calls, raw_tool_calls=raw, usage=usage)


def _iter_stream_json(resp: Any) -> Iterator[dict[str, Any]]:
    buf = b""
    while True:
        chunk = resp.read(512)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if not text or text == "[DONE]":
                continue
            if text.startswith("data:"):
                text = text[5:].strip()
            if text == "[DONE]":
                return
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def detect_llm(*, local: bool = False) -> OpenAICompatLLM:
    """Pick a live backend from the environment. Raises if nothing is configured."""
    if local:
        ollama = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = urlparse(ollama).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("offline/--local: OLLAMA_HOST must be loopback (127.0.0.1)")
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        if not _ollama_up(ollama):
            raise RuntimeError(
                "offline/--local: Ollama is not running. Start it with:\n"
                "  brew services start ollama && ollama pull qwen2.5:3b"
            )
        return OpenAICompatLLM(
            base_url=f"{ollama.rstrip('/')}/v1",
            model=model,
            api_key="ollama",
            local_only=True,
        )

    azure_ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_dep = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    if azure_ep and azure_key and azure_dep:
        return OpenAICompatLLM(
            base_url=f"{azure_ep}/openai/deployments/{azure_dep}",
            model=azure_dep,
            api_key=azure_key,
            extra_headers={"api-key": azure_key},
        )

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAICompatLLM(base_url=base, model=model, api_key=openai_key)

    github = os.environ.get("GITHUB_TOKEN", "")
    if github:
        return OpenAICompatLLM(
            base_url="https://models.inference.ai.azure.com",
            model=os.environ.get("GITHUB_MODEL", "gpt-4o-mini"),
            api_key=github,
        )

    ollama = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    if _ollama_up(ollama):
        return OpenAICompatLLM(base_url=f"{ollama.rstrip('/')}/v1", model=model, api_key="ollama")

    raise RuntimeError(
        "No LLM backend found. This agent is model-driven — start one:\n"
        "  brew services start ollama && ollama pull qwen2.5:3b\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "  export AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... AZURE_OPENAI_DEPLOYMENT=..."
    )


def _ollama_up(host: str) -> bool:
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
