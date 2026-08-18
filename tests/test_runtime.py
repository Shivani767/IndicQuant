"""Live LLM-driven agent loop — not a hardcoded policy."""

from __future__ import annotations

from indicquant.agent.env import AgentEnv
from indicquant.agent.llm import LLMOutput, ScriptedLLM
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent
from indicquant.agent.tools import user_tools
from indicquant.agent.types import ToolCall


def _agent(script: list[LLMOutput], *, auto_confirm: bool = True) -> Agent:
    store = EpisodicStore()
    return Agent(
        ScriptedLLM(script),
        AgentEnv(tools=user_tools(store)),
        memory=store,
        auto_confirm=auto_confirm,
    )


def _tool(turn) -> object:
    return next(e for e in turn.events if e.kind == "tool")


def test_agent_calls_tool_then_answers() -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[ToolCall("calculator", {"expr": "500 + 500*18/100"})],
        ),
        LLMOutput(content="कुल राशि 590 रुपये है।"),
    ]
    turn = _agent(script).act("500 रुपये पर 18 प्रतिशत कर कितना लगेगा?")
    tools = [e for e in turn.events if e.kind == "tool"]
    assert tools
    assert "590" in tools[0].text
    assert turn.reply == "कुल राशि 590 रुपये है।"
    assert turn.messages[-1]["role"] == "assistant"
    assert turn.grounded is True
    assert turn.messages[-1]["role"] == "assistant"


def test_agent_binds_native_city_before_tool() -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[
                ToolCall("book_ticket", {"origin": "मुंबई", "destination": "दिल्ली", "passengers": 2})
            ],
        ),
        LLMOutput(content="टिकट बुक हो गई।"),
    ]
    turn = _agent(script).act("मुंबई से दिल्ली के लिए 2 टिकट बुक करो")
    bind = next(e for e in turn.events if e.kind == "bind")
    assert bind.data["raw"]["args"]["origin"] == "मुंबई"
    assert bind.data["bound"]["args"]["origin"] == "BOM"
    payload = _tool(turn).data
    assert payload["call"]["args"]["origin"] == "BOM"
    assert payload["result"]["ok"] is True
    assert "BOMDEL" in payload["result"]["output"]
    assert "टिकट" in turn.reply


def test_agent_memory_survives_a_turn() -> None:
    script = [
        LLMOutput(content="", tool_calls=[ToolCall("remember", {"note": "मेरा नाम शिवानी है"})]),
        LLMOutput(content="याद रखा।"),
        LLMOutput(content="", tool_calls=[ToolCall("recall", {"query": "नाम"})]),
        LLMOutput(content="आपका नाम शिवानी है।"),
    ]
    agent = _agent(script)
    first = agent.act("मेरा नाम शिवानी है, याद रखना")
    assert _tool(first).kind == "tool"
    second = agent.act("मेरा नाम क्या है?")
    assert "शिवानी" in _tool(second).text
    assert "शिवानी" in second.reply
    assert agent.memory.notes == ["मेरा नाम शिवानी है"]


def test_multi_turn_history_is_kept() -> None:
    script = [
        LLMOutput(content="नमस्ते!"),
        LLMOutput(content="हाँ, मैं यहाँ हूँ।"),
    ]
    agent = _agent(script)
    agent.act("नमस्ते")
    agent.act("अभी भी याद है?")
    roles = [m["role"] for m in agent.messages]
    assert roles.count("user") == 2
    assert roles[0] == "system"


def test_booking_waits_for_confirm() -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[
                ToolCall("book_ticket", {"origin": "मुंबई", "destination": "दिल्ली", "passengers": 2})
            ],
        ),
        LLMOutput(content="टिकट बुक हो गई।"),
    ]
    agent = _agent(script, auto_confirm=False)
    paused = agent.act("मुंबई से दिल्ली के लिए 2 टिकट बुक करो")
    assert paused.pending is not None
    assert "BOM" in paused.pending["preview"]
    assert "uv run" not in paused.reply
    assert "मुंबई" in paused.reply
    assert "दिल्ली" in paused.reply
    assert "uv run" not in paused.reply
    assert "मुंबई" in paused.reply or "Mumbai" in paused.reply
    assert not any(e.kind == "tool" for e in paused.events)
    done = agent.confirm()
    assert _tool(done).data["result"]["ok"] is True
    assert "टिकट" in done.reply
    assert done.pending is None


def test_parallel_independent_tools() -> None:
    script = [
        LLMOutput(
            content="",
            tool_calls=[
                ToolCall("calculator", {"expr": "1+1"}),
                ToolCall("calculator", {"expr": "2+2"}),
            ],
        ),
        LLMOutput(content="2 और 4"),
    ]
    turn = _agent(script).act("1+1 और 2+2")
    tools = [e for e in turn.events if e.kind == "tool"]
    assert len(tools) == 2
    outputs = {e.data["result"]["output"] for e in tools}
    assert "2.0" in outputs
    assert "4.0" in outputs


def test_inr_gst_binds_native_amount() -> None:
    script = [
        LLMOutput(content="", tool_calls=[ToolCall("inr_gst", {"amount": "५००", "rate": "18"})]),
        LLMOutput(content="कुल 590 रुपये।"),
    ]
    turn = _agent(script).act("500 पर GST")
    payload = _tool(turn).data
    assert payload["call"]["args"]["amount"] == 500.0
    assert payload["result"]["data"]["total"] == 590.0
    assert turn.grounded is True


def test_open_question_needs_no_tool() -> None:
    script = [LLMOutput(content="प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे सूर्य के प्रकाश से भोजन बनाते हैं।")]
    turn = _agent(script).act("प्रकाश संश्लेषण क्या है?")
    assert not any(e.kind == "tool" for e in turn.events)
    assert "प्रकाश" in turn.reply


def test_lookup_miss_still_lets_model_answer() -> None:
    script = [
        LLMOutput(content="", tool_calls=[ToolCall("lookup", {"query": "quantum foam"})]),
        LLMOutput(content="It is a way some physicists describe spacetime at tiny scales."),
    ]
    turn = _agent(script).act("What is quantum foam?")
    payload = _tool(turn).data
    assert payload["result"]["ok"] is True
    assert "no local KB" in payload["result"]["output"]
    assert "spacetime" in turn.reply
