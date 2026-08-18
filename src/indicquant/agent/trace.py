"""Structured terminal traces from what actually happened — not from a classifier."""

from __future__ import annotations

from indicquant.agent.runtime import AgentTurn


def format_stages(turn: AgentTurn, *, verbose: bool = False) -> list[str]:
    lines: list[str] = []
    for event in turn.events:
        if event.kind == "bind":
            bound = (event.data.get("bound") or {}) if event.data else {}
            name = bound.get("name") or ""
            params = bound.get("args") or {}
            lines.append(f"[bind]  tool={name}, params={params}")
            if verbose and event.data.get("raw"):
                lines.append(f"[bind]    raw={event.data['raw'].get('args')}")
        elif event.kind == "tool":
            call = (event.data.get("call") or {}) if event.data else {}
            name = call.get("name") or "tool"
            ms = event.data.get("duration_ms")
            suffix = f" {ms:.0f}ms" if isinstance(ms, (int, float)) else ""
            lines.append(f"[act]   calling {name}...{suffix}")
            if verbose:
                result = event.data.get("result") or {}
                lines.append(f"[act]    output={result.get('output')}")
        elif event.kind == "critic":
            lines.append(f"[critic] {event.text}")
        elif event.kind == "confirm":
            lines.append(f"[pending] {event.text}")
        elif event.kind == "retry":
            lines.append(f"[retry] {event.text}")

    verdict = turn.verdict or {}
    v = int(verdict.get("verified") or 0)
    t = int(verdict.get("total") or 0)
    if t:
        lines.append(f"[ground] verified {v}/{t} claims against tool fields")
    elif any(e.kind == "tool" for e in turn.events):
        lines.append("[ground] no precise claims to check")
    if turn.critic_rejections:
        lines.append(f"[critic] rejected {turn.critic_rejections} unverified claims")
    if turn.pending_id:
        lines.append(f"[pending] waiting for confirm {turn.pending_id}")

    if verbose and turn.memory_diff:
        added = turn.memory_diff.get("added") or []
        removed = turn.memory_diff.get("removed") or []
        if added or removed:
            lines.append(f"[memory] +{len(added)} / -{len(removed)}")
            for fact in added:
                lines.append(f"[memory]   + {fact}")
            for fact in removed:
                lines.append(f"[memory]   - {fact}")
    return lines
