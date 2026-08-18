"""Deterministic grounding critic — no LLM-as-judge."""

from __future__ import annotations

from indicquant.agent.env import AgentEnv
from indicquant.agent.grounding import extract_claims, ground_reply
from indicquant.agent.llm import LLMOutput, ScriptedLLM
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent
from indicquant.agent.tools import user_tools
from indicquant.agent.types import ToolCall


def test_extract_skips_small_counts() -> None:
    claims = extract_claims("2 tickets, 18% GST, total 590, PNR BOMDEL02")
    assert "590" in claims
    assert "BOMDEL02" in claims
    assert "2" not in claims
    assert "18" not in claims


def test_ground_ok_when_number_in_tool_output() -> None:
    report = ground_reply("कुल 590 रुपये", ["base=500 gst=90 total=590"])
    assert report.ok is True
    assert report.missing == []


def test_ground_flags_invented_fare() -> None:
    report = ground_reply("कुल 999 रुपये", ["base=500 gst=90 total=590"])
    assert report.ok is False
    assert "999" in report.missing


def test_chitchat_without_tools_is_skipped() -> None:
    assert ground_reply("नमस्ते", []).skipped is True


def test_fare_does_not_match_superset_number() -> None:
    report = ground_reply("कुल 590 रुपये", ["total=1590"])
    assert report.ok is False
    assert "590" in report.missing


def test_field_match_is_exact() -> None:
    report = ground_reply("PNR BOMDEL02 total 590", [{"output": "booked", "data": {"pnr": "BOMDEL02", "total": 590.0}}])
    assert report.ok is True
    assert report.verified_n == 2


def test_runtime_rewrites_hallucinated_total() -> None:
    script = [
        LLMOutput(content="", tool_calls=[ToolCall("calculator", {"expr": "500 + 90"})]),
        LLMOutput(content="कुल 999 रुपये है।"),
        LLMOutput(content="कुल राशि 590 रुपये है।"),
    ]
    agent = Agent(ScriptedLLM(script), AgentEnv(tools=user_tools(EpisodicStore())))
    turn = agent.act("500 पर 90 जोड़ो")
    assert "590" in turn.reply
    assert "999" not in turn.reply
    assert any(e.kind == "critic" for e in turn.events)
    assert turn.grounded is True
    rewritten = [e for e in turn.events if e.kind == "reply" and e.data.get("rewritten")]
    assert rewritten
