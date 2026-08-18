"""Provenance memory does not reuse inferred-unverified facts."""

from __future__ import annotations

from indicquant.agent.llm import OpenAICompatLLM
from indicquant.agent.memory import EpisodicStore


def test_inferred_unverified_is_not_recalled() -> None:
    store = EpisodicStore()
    store.remember("home city is मुंबई", source="inferred", verified=False)
    store.remember("मेरा नाम शिवानी है", source="user_stated", verified=False)
    store.remember("PNR BOMDEL02", source="tool_call", verified=True, tool="book_ticket")
    assert "home city is मुंबई" not in store.recall()
    assert "मेरा नाम शिवानी है" in store.recall()
    assert "PNR BOMDEL02" in store.recall()


def test_memory_diff() -> None:
    store = EpisodicStore()
    before = store.snapshot()
    store.remember("x", source="user_stated")
    diff = store.diff(before)
    assert diff["added"] == ["x"]


def test_local_only_blocks_cloud_url() -> None:
    llm = OpenAICompatLLM(base_url="https://api.openai.com/v1", model="x", local_only=True)
    try:
        llm.complete([], [])
        raise AssertionError("should block")
    except RuntimeError as exc:
        assert "offline" in str(exc)
