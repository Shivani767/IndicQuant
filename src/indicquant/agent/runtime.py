"""The agent core: the model decides every action.

Loop: user → LLM → (bind → tool → observation)* → verify → reply.

Nothing here classifies the user's intent. Locale bind runs only after the model
has already chosen a tool. The verifier runs only after the model has already
written an answer.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from indicquant.agent.adapter import BoundaryAdapter, detect_language
from indicquant.agent.env import AgentEnv
from indicquant.agent.grounding import critic_prompt, ground_reply, redact_claims
from indicquant.agent.llm import LLM, LLMOutput
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.pending import PendingAction, PendingStore
from indicquant.agent.schema import openai_tools
from indicquant.agent.tools import city_label
from indicquant.agent.types import ToolCall, ToolResult

Emit = Callable[[dict[str, Any]], None]

SYSTEM = """You are a general assistant, like ChatGPT or Claude. Answer ANY question the user asks: explanations, coding, math, language, culture, current topics, advice, or casual chat. Match their language and script.

Tools are optional. Most questions need none.
- Do not call a tool just because one exists. Never say you can only book tickets or only use tools.
- calculator / inr_gst: only for exact arithmetic the user asked you to compute.
- search: only if you need a specific fact you are unsure of. If search misses, still answer from knowledge.
- lookup: tiny local notes (GST, a few cities). If it misses, answer yourself.
- book_ticket: only when they ask to book tickets. Keep origin → destination; copy PNR from the tool.
- remember / recall / now: only when they ask to remember, recall, or need the time.

Never dump tool JSON, IATA rewrites, or shell commands. Never invent a PNR, GST total, or clock time — copy those from tools.
"""

_STATUS = {
    "hi": {"think": "सोच रहा हूँ…", "work": "काम चल रहा है…", "write": "जवाब लिख रहा हूँ…"},
    "ta": {"think": "யோசிக்கிறேன்…", "work": "செயல்படுத்துகிறேன்…", "write": "பதில் எழுதுகிறேன்…"},
    "en": {"think": "Thinking…", "work": "Working…", "write": "Writing a reply…"},
}

_TOOL_STATUS = {
    "book_ticket": {"en": "Preparing your booking…", "hi": "टिकट तैयार कर रहा हूँ…", "ta": "டிக்கெட்டை தயார் செய்கிறேன்…"},
    "inr_gst": {"en": "Calculating GST…", "hi": "GST निकाल रहा हूँ…", "ta": "GST கணக்கிடுகிறேன்…"},
    "calculator": {"en": "Calculating…", "hi": "हिसाब लगा रहा हूँ…", "ta": "கணக்கிடுகிறேன்…"},
    "lookup": {"en": "Looking that up…", "hi": "ढूँढ रहा हूँ…", "ta": "தேடுகிறேன்…"},
    "search": {"en": "Searching…", "hi": "खोज रहा हूँ…", "ta": "தேடுகிறேன்…"},
    "remember": {"en": "Saving that…", "hi": "याद रख रहा हूँ…", "ta": "சேமிக்கிறேன்…"},
    "recall": {"en": "Checking memory…", "hi": "याद देख रहा हूँ…", "ta": "நினைவில் பார்க்கிறேன்…"},
}

SIDE_EFFECTS = frozenset({"book_ticket"})
SERIAL_TOOLS = frozenset({"remember", "recall", "book_ticket", "type_text", "click"})


class TurnLog(Protocol):
    def append(self, turn: AgentTurn) -> None: ...


@dataclass
class TraceEvent:
    kind: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    reply: str
    events: list[TraceEvent]
    messages: list[dict[str, Any]]
    turn_id: str = ""
    duration_ms: float = 0.0
    grounded: bool = True
    missing_claims: list[str] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    user: str = ""
    critic_rejections: int = 0
    verdict: dict[str, Any] = field(default_factory=dict)
    pending_id: str | None = None
    memory_diff: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user": self.user,
            "reply": self.reply,
            "duration_ms": round(self.duration_ms, 1),
            "grounded": self.grounded,
            "missing_claims": self.missing_claims,
            "pending": self.pending,
            "pending_id": self.pending_id,
            "usage": self.usage,
            "critic_rejections": self.critic_rejections,
            "verdict": self.verdict,
            "memory_diff": self.memory_diff,
            "events": [{"kind": e.kind, "text": e.text, "data": e.data} for e in self.events],
        }


class Agent:
    def __init__(
        self,
        llm: LLM,
        env: AgentEnv,
        adapter: BoundaryAdapter | None = None,
        *,
        max_steps: int = 8,
        system: str = SYSTEM,
        auto_confirm: bool = True,
        memory: EpisodicStore | None = None,
        log: TurnLog | None = None,
        session_id: str | None = None,
        pending_store: PendingStore | None = None,
        local: bool = False,
    ) -> None:
        self.llm = llm
        self.env = env
        self.adapter = adapter or BoundaryAdapter()
        self.max_steps = max_steps
        self.system = system
        self.auto_confirm = auto_confirm
        self.memory = memory or EpisodicStore()
        self.log = log
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.pending_store = pending_store if pending_store is not None else PendingStore()
        self.local = local
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._pending: dict[str, Any] | None = None
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.critic_rejections = 0

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        return openai_tools(self.env.tools)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}]
        self._pending = None
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def act(self, user_text: str, emit: Emit | None = None) -> AgentTurn:
        t0 = time.perf_counter()
        before = self.memory.snapshot()
        if self._pending:
            pid = self._pending.get("action_id")
            if pid:
                try:
                    self.pending_store.mark(pid, "cancelled")
                except (KeyError, ValueError):
                    pass
            self._pending = None
        self.messages.append({"role": "user", "content": user_text})
        turn = self._loop([], t0, emit)
        turn.user = user_text
        turn.memory_diff = self.memory.diff(before)
        if self.log:
            self.log.append(turn)
        return turn

    def confirm(self, action_id: str | None = None, emit: Emit | None = None) -> AgentTurn:
        t0 = time.perf_counter()
        if action_id and (self._pending is None or self._pending.get("action_id") != action_id):
            self.restore_pending(self.pending_store.get(action_id))
        pending = self._pending
        if pending is None:
            return AgentTurn(
                reply="Nothing is waiting for confirmation.",
                events=[],
                messages=list(self.messages),
            )
        events = [TraceEvent("confirmed", pending["preview"], {"preview": pending["preview"]})]
        self.messages.append(
            {
                "role": "assistant",
                "content": getattr(pending.get("out"), "content", None) or None,
                "tool_calls": pending["assistant_tc"],
            }
        )
        lang = _lang_of(self.messages)
        _emit(emit, "status", text=_tool_status("book_ticket", lang))
        for item in pending["queued"]:
            self._commit_call(item["raw"], item["bound"], item["cid"], events)
        pid = pending.get("action_id")
        if pid:
            try:
                self.pending_store.mark(pid, "confirmed")
            except (KeyError, ValueError):
                pass
        self._pending = None
        turn = self._loop(events, t0, emit)
        turn.user = "[confirm]"
        if self.log:
            self.log.append(turn)
        return turn

    def cancel(self, action_id: str | None = None) -> AgentTurn:
        pid = action_id or (self._pending or {}).get("action_id")
        if pid:
            try:
                self.pending_store.mark(pid, "cancelled")
            except (KeyError, ValueError):
                pass
        self._pending = None
        return AgentTurn(
            reply=f"Cancelled {pid or 'pending action'}.",
            events=[TraceEvent("cancelled", str(pid or ""), {})],
            messages=list(self.messages),
        )

    def restore_pending(self, action: PendingAction) -> None:
        if action.refresh_status() != "pending":
            raise ValueError(f"{action.id} is {action.status}")
        if action.messages:
            self.messages = list(action.messages)
        self._pending = {
            "out": LLMOutput(content=""),
            "assistant_tc": action.assistant_tc,
            "queued": [_queued_from_dict(item) for item in action.queued],
            "preview": action.preview,
            "action_id": action.id,
        }

    def _loop(self, events: list[TraceEvent], t0: float, emit: Emit | None = None) -> AgentTurn:
        lang = _lang_of(self.messages)
        _emit(emit, "status", text=_status("think", lang))
        for _ in range(self.max_steps):
            out = self._complete(emit, stream_tokens=True)
            self._add_usage(getattr(out, "usage", None))
            if not out.tool_calls:
                return self._finalize(out.content or "", events, t0, emit, streamed=True)

            assistant_tc = _assistant_tool_calls(out.tool_calls, out.raw_tool_calls)
            queued, blocked = self._prepare(out.tool_calls, assistant_tc)

            finish = next((q for q in queued if q["bound"].name == "finish"), None)
            if finish is not None:
                reply = str(finish["bound"].args.get("answer", "") or out.content or "")
                return self._finalize(reply, events, t0, emit)

            if blocked is not None:
                return self._pause(out, assistant_tc, queued, blocked, events, t0, emit)

            name = queued[0]["bound"].name if queued else "tool"
            _emit(emit, "status", text=_tool_status(name, lang))
            self.messages.append(
                {
                    "role": "assistant",
                    "content": out.content or None,
                    "tool_calls": assistant_tc,
                }
            )
            self._run_queued(queued, events)
            _emit(emit, "status", text=_status("write", lang))

        return self._finalize("I ran out of tool steps before I could finish.", events, t0, emit)

    def _complete(self, emit: Emit | None, *, stream_tokens: bool) -> LLMOutput:
        on_delta = None
        if stream_tokens and emit is not None:

            def _delta(text: str) -> None:
                _emit(emit, "token", text=text)

            on_delta = _delta
        stream = getattr(self.llm, "stream_complete", None)
        if callable(stream):
            return stream(self.messages, self.tools_schema, on_delta=on_delta)
        return self.llm.complete(self.messages, self.tools_schema)

    def _pause(
        self,
        out: LLMOutput,
        assistant_tc: list[dict[str, Any]],
        queued: list[dict[str, Any]],
        blocked: dict[str, Any],
        events: list[TraceEvent],
        t0: float,
        emit: Emit | None = None,
    ) -> AgentTurn:
        preview = _preview(blocked["bound"])
        queued_all = queued + [blocked]
        action = self.pending_store.create(
            session_id=self.session_id,
            tool=blocked["bound"].name,
            args=dict(blocked["bound"].args),
            preview=preview,
            intent={},
            assistant_tc=assistant_tc,
            queued=[_queued_to_dict(item) for item in queued_all],
            messages=list(self.messages),
            raw=blocked["raw"].to_dict(),
        )
        self._pending = {
            "out": out,
            "assistant_tc": assistant_tc,
            "queued": queued_all,
            "preview": preview,
            "action_id": action.id,
        }
        events.append(
            TraceEvent(
                "bind",
                _bind_text(blocked["raw"], blocked["bound"]),
                {"raw": blocked["raw"].to_dict(), "bound": blocked["bound"].to_dict()},
            )
        )
        events.append(
            TraceEvent(
                "confirm",
                preview,
                {
                    "tool": blocked["bound"].name,
                    "args": blocked["bound"].args,
                    "preview": preview,
                    "id": action.id,
                },
            )
        )
        user_text = _last_user(self.messages)
        lang = detect_language(user_text)
        origin_label = city_label(
            str(blocked["raw"].args.get("origin") or blocked["bound"].args.get("origin") or ""),
            lang,
        )
        dest_label = city_label(
            str(blocked["raw"].args.get("destination") or blocked["bound"].args.get("destination") or ""),
            lang,
        )
        passengers = int(blocked["bound"].args.get("passengers") or 1)
        reply = _pause_reply(lang, origin_label, dest_label, passengers)
        pending = {
            "id": action.id,
            "tool": blocked["bound"].name,
            "args": blocked["bound"].args,
            "raw": blocked["raw"].to_dict(),
            "preview": preview,
            "origin_label": origin_label,
            "destination_label": dest_label,
            "passengers": passengers,
        }
        _emit(emit, "pending", **pending)
        _emit(emit, "reply", text=reply)
        return AgentTurn(
            reply=reply,
            events=events,
            messages=list(self.messages),
            turn_id=uuid.uuid4().hex[:10],
            duration_ms=(time.perf_counter() - t0) * 1000,
            grounded=True,
            pending=pending,
            pending_id=action.id,
            critic_rejections=self.critic_rejections,
            usage=dict(self._usage),
        )

    def _prepare(
        self, calls: list[ToolCall], assistant_tc: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        queued: list[dict[str, Any]] = []
        blocked: dict[str, Any] | None = None
        for i, call in enumerate(calls):
            bound = self.adapter.rewrite_tool_call(call)
            item = {"raw": call, "bound": bound, "cid": assistant_tc[i]["id"]}
            gate = (
                bound.name in SIDE_EFFECTS
                and not self.auto_confirm
                and not _truthy(bound.args.get("confirmed"))
            )
            if gate and blocked is None:
                blocked = item
            else:
                queued.append(item)
        return queued, blocked

    def _run_queued(self, queued: list[dict[str, Any]], events: list[TraceEvent]) -> None:
        serial = any(item["bound"].name in SERIAL_TOOLS for item in queued)
        if serial or len(queued) <= 1:
            for item in queued:
                self._commit_call(item["raw"], item["bound"], item["cid"], events)
            return

        def _exec(item: dict[str, Any]) -> tuple[dict[str, Any], ToolResult, bool, float]:
            started = time.perf_counter()
            result, retried = self._step_with_retry(item["bound"])
            return item, result, retried, (time.perf_counter() - started) * 1000

        with ThreadPoolExecutor(max_workers=min(4, len(queued))) as pool:
            for item, result, retried, ms in pool.map(_exec, queued):
                if retried:
                    events.append(TraceEvent("retry", f"{item['bound'].name} retried", {}))
                self._record(item["raw"], item["bound"], item["cid"], result, events, duration_ms=ms)

    def _commit_call(
        self, raw: ToolCall, bound: ToolCall, cid: str, events: list[TraceEvent]
    ) -> ToolResult:
        started = time.perf_counter()
        result, retried = self._step_with_retry(bound)
        ms = (time.perf_counter() - started) * 1000
        if retried:
            events.append(TraceEvent("retry", f"{bound.name} retried", {}))
        self._record(raw, bound, cid, result, events, duration_ms=ms)
        return result

    def _step_with_retry(self, bound: ToolCall) -> tuple[ToolResult, bool]:
        bound = _strip_runtime_args(bound)
        result = self.env.step(bound)
        if result.ok:
            return result, False
        transient = any(tok in result.output.lower() for tok in ("timeout", "unavailable", "temporar"))
        if not transient:
            return result, False
        return self.env.step(bound), True

    def _record(
        self,
        raw: ToolCall,
        bound: ToolCall,
        cid: str,
        result: ToolResult,
        events: list[TraceEvent],
        *,
        duration_ms: float = 0.0,
    ) -> None:
        events.append(
            TraceEvent(
                "bind",
                _bind_text(raw, bound),
                {"raw": raw.to_dict(), "bound": bound.to_dict()},
            )
        )
        events.append(
            TraceEvent(
                "tool",
                f"{bound.name}({bound.args}) -> {result.output}",
                {
                    "call": bound.to_dict(),
                    "raw": raw.to_dict(),
                    "result": result.to_dict(),
                    "duration_ms": round(duration_ms, 1),
                },
            )
        )
        if bound.name == "book_ticket" and result.ok:
            pnr = (result.data or {}).get("pnr")
            if pnr:
                self.memory.remember(
                    f"PNR {pnr} {result.data.get('origin')}->{result.data.get('destination')}",
                    source="tool_call",
                    verified=True,
                    tool="book_ticket",
                )
        self.messages.append(_tool_msg(cid, bound, result))

    def _finalize(
        self,
        reply: str,
        events: list[TraceEvent],
        t0: float,
        emit: Emit | None = None,
        *,
        streamed: bool = False,
    ) -> AgentTurn:
        events.append(TraceEvent("reply", reply))
        self.messages.append({"role": "assistant", "content": reply})
        results = [e.data["result"] for e in events if e.kind == "tool" and "result" in e.data]
        report = ground_reply(reply, results)
        if results and not report.ok:
            self.critic_rejections += len(report.rejected)
            events.append(
                TraceEvent("critic", f"ungrounded: {', '.join(report.missing)}", report.to_dict())
            )
            self.messages.append({"role": "user", "content": critic_prompt(report)})
            lang = _lang_of(self.messages)
            _emit(emit, "status", text=_status("write", lang))
            out = self._complete(emit, stream_tokens=True)
            self._add_usage(getattr(out, "usage", None))
            reply = out.content or reply if not out.tool_calls else (out.content or reply)
            streamed = True
            self.messages.append({"role": "assistant", "content": reply})
            events.append(TraceEvent("reply", reply, {"rewritten": True}))
            report = ground_reply(reply, results)
            if not report.ok:
                reply = redact_claims(reply, report.rejected)
                events.append(TraceEvent("reply", reply, {"redacted": True}))
                self.messages[-1] = {"role": "assistant", "content": reply}
                report = ground_reply(reply, results)
                streamed = False
        if not streamed:
            _emit(emit, "token", text=reply)
        _emit(emit, "reply", text=reply)
        return AgentTurn(
            reply=reply,
            events=events,
            messages=list(self.messages),
            turn_id=uuid.uuid4().hex[:10],
            duration_ms=(time.perf_counter() - t0) * 1000,
            grounded=report.ok,
            missing_claims=list(report.missing),
            usage=dict(self._usage),
            critic_rejections=self.critic_rejections,
            verdict=report.to_dict(),
        )

    def _add_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self._usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)


def _emit(emit: Emit | None, kind: str, **data: Any) -> None:
    if emit is None:
        return
    emit({"type": kind, **data})


def _last_user(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "")
        if text.startswith("["):
            continue
        return text
    return ""


def _lang_of(messages: list[dict[str, Any]]) -> str:
    return detect_language(_last_user(messages))


def _status(kind: str, lang: str) -> str:
    key = "hi" if lang == "hi_en" else lang
    row = _STATUS.get(key) or _STATUS["en"]
    return row.get(kind) or _STATUS["en"][kind]


def _tool_status(name: str, lang: str) -> str:
    key = "hi" if lang == "hi_en" else lang
    row = _TOOL_STATUS.get(name) or {}
    return row.get(key) or row.get("en") or _status("work", lang)


def _pause_reply(lang: str, origin: str, dest: str, passengers: int) -> str:
    n = passengers
    if lang in {"hi", "hi_en"}:
        return f"{n} टिकट {origin} से {dest} के लिए तैयार हैं। पुष्टि करें तो बुक कर दूँ।"
    if lang == "ta":
        return f"{origin} இலிருந்து {dest} க்கு {n} டிக்கெட் தயார். உறுதி செய்தால் பதிவு செய்கிறேன்."
    return f"Ready to book {n} ticket(s) from {origin} to {dest}. Confirm to continue."


def _bind_text(raw: ToolCall, bound: ToolCall) -> str:
    if raw.args == bound.args:
        return f"{bound.name}: args already canonical"
    return f"{raw.args} → {bound.args}"


def _preview(call: ToolCall) -> str:
    if call.name == "book_ticket":
        return f"{call.args.get('passengers', 1)} × {call.args.get('origin')} → {call.args.get('destination')}"
    return f"{call.name}({call.args})"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _strip_runtime_args(call: ToolCall) -> ToolCall:
    if "confirmed" not in call.args:
        return call
    args = {k: v for k, v in call.args.items() if k != "confirmed"}
    return ToolCall(call.name, args, thought=call.thought)


def _queued_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {"raw": item["raw"].to_dict(), "bound": item["bound"].to_dict(), "cid": item["cid"]}


def _queued_from_dict(item: dict[str, Any]) -> dict[str, Any]:
    raw, bound = item["raw"], item["bound"]
    return {
        "raw": ToolCall(raw["name"], raw["args"], thought=raw.get("thought") or ""),
        "bound": ToolCall(bound["name"], bound["args"], thought=bound.get("thought") or ""),
        "cid": item["cid"],
    }


def _assistant_tool_calls(
    calls: list[ToolCall], raw: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for i, call in enumerate(calls):
        cid = str((raw[i].get("id") if i < len(raw) else None) or f"call_{uuid.uuid4().hex[:12]}")
        out.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)},
            }
        )
    return out


def _tool_msg(call_id: str, call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": call.name,
        "content": result.output if result.ok else f"ERROR: {result.output}",
    }
