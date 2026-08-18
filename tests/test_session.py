"""Session traces persist to JSONL."""

from __future__ import annotations

from indicquant.agent.env import AgentEnv
from indicquant.agent.llm import LLMOutput, ScriptedLLM
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent
from indicquant.agent.session import SessionHub, load_trace_files
from indicquant.agent.tools import user_tools


def test_session_jsonl_and_title(tmp_path) -> None:
    script = [LLMOutput(content="नमस्ते!")]

    def make() -> Agent:
        return Agent(ScriptedLLM(list(script)), AgentEnv(tools=user_tools(EpisodicStore())))

    hub = SessionHub(make, tmp_path)
    sess = hub.create()
    turn = sess.agent.act("नमस्ते")
    hub.record(sess, "नमस्ते", turn)
    path = tmp_path / f"{sess.id}.jsonl"
    assert path.exists()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert sess.title == "नमस्ते"
    rows = load_trace_files(tmp_path)
    assert rows[0]["id"] == sess.id
    assert rows[0]["turns"] == 1
