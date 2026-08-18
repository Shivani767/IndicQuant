"""Narrow, honest eval: Indic booking queries with known-correct answers.

Runs against a scripted LLM so the number is reproducible without a GPU.
Optional --live uses the configured model (not the published score).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indicquant.agent.env import AgentEnv
from indicquant.agent.llm import LLMOutput, ScriptedLLM
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent
from indicquant.agent.tools import user_tools
from indicquant.agent.types import ToolCall

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "evals" / "indic_booking.json"


def _cases_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "evals" / "indic_booking.json"
        if candidate.exists():
            return candidate
    return DEFAULT_CASES


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _cases_path()
    return json.loads(target.read_text(encoding="utf-8"))


def _script(case: dict[str, Any]) -> list[LLMOutput]:
    out: list[LLMOutput] = []
    for step in case.get("script") or []:
        if "tool" in step:
            out.append(
                LLMOutput(
                    content="",
                    tool_calls=[ToolCall(str(step["tool"]), dict(step.get("args") or {}))],
                )
            )
        else:
            out.append(LLMOutput(content=str(step.get("reply") or "")))
    return out


@dataclass
class CaseScore:
    id: str
    ok: bool
    grounded: bool
    hallucination_caught: bool
    duration_ms: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "grounded": self.grounded,
            "hallucination_caught": self.hallucination_caught,
            "duration_ms": round(self.duration_ms, 1),
            "errors": self.errors,
        }


def score_case(case: dict[str, Any], turn: Any) -> CaseScore:
    expect = case.get("expect") or {}
    errors: list[str] = []
    caught = any(e.kind == "critic" for e in turn.events)
    if expect.get("hallucination_caught") and not caught:
        errors.append("expected critic to catch a hallucination")
    if expect.get("grounded") is True and not turn.grounded:
        errors.append("expected grounded reply")
    if expect.get("grounded") is False and turn.grounded:
        errors.append("expected ungrounded reply")
    for needle in expect.get("reply_contains") or []:
        if needle not in (turn.reply or ""):
            errors.append(f"reply missing {needle!r}")
    for needle in expect.get("reply_excludes") or []:
        if needle in (turn.reply or ""):
            errors.append(f"reply still contains {needle!r}")
    bind_origin = expect.get("bind_origin")
    bind_dest = expect.get("bind_dest")
    if bind_origin or bind_dest:
        tools = [e for e in turn.events if e.kind == "tool"]
        if not tools:
            errors.append("no tool call")
        else:
            args = tools[0].data["call"]["args"]
            if bind_origin and args.get("origin") != bind_origin:
                errors.append(f"origin {args.get('origin')} != {bind_origin}")
            if bind_dest and args.get("destination") != bind_dest:
                errors.append(f"dest {args.get('destination')} != {bind_dest}")
    return CaseScore(
        id=str(case.get("id")),
        ok=not errors,
        grounded=bool(turn.grounded),
        hallucination_caught=caught,
        duration_ms=float(turn.duration_ms or 0),
        errors=errors,
    )


def run_eval(path: Path | None = None) -> dict[str, Any]:
    cases = load_cases(path)
    scores: list[CaseScore] = []
    for case in cases:
        store = EpisodicStore()
        agent = Agent(
            ScriptedLLM(_script(case)),
            AgentEnv(tools=user_tools(store)),
            memory=store,
            auto_confirm=True,
        )
        turn = agent.act(str(case["query"]))
        scores.append(score_case(case, turn))
    n = len(scores) or 1
    completed = sum(1 for s in scores if s.ok)
    # Published hallucination *rate* = critic-caught / cases that tried to invent
    invented = [c for c in cases if (c.get("expect") or {}).get("hallucination_caught")]
    caught = sum(1 for s in scores if s.hallucination_caught and s.id in {c["id"] for c in invented})
    return {
        "cases": len(scores),
        "task_completion_rate": round(completed / n, 3),
        "hallucination_catch_rate": round(caught / len(invented), 3) if invented else None,
        "latency_ms_mean": round(sum(s.duration_ms for s in scores) / n, 1),
        "scores": [s.to_dict() for s in scores],
    }
