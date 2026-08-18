"""Deterministic grounding critic — thin wrapper over the hard verifier."""

from __future__ import annotations

from typing import Any

from indicquant.agent.verifier import Verdict as GroundingReport
from indicquant.agent.verifier import critic_prompt, extract_claims, redact_claims, verify_reply


def ground_reply(reply: str, tool_outputs: list[Any]) -> GroundingReport:
    """Accept raw output strings (tests) or result dicts (runtime)."""
    results: list[dict[str, Any]] = []
    for item in tool_outputs:
        if isinstance(item, str):
            results.append({"output": item, "data": {}})
        elif isinstance(item, dict):
            results.append(item)
        else:
            results.append({"output": str(item), "data": {}})
    return verify_reply(reply, results)


__all__ = [
    "GroundingReport",
    "critic_prompt",
    "extract_claims",
    "ground_reply",
    "redact_claims",
]
