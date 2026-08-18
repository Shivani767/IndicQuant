"""Persisted turn traces. Every live session is a JSONL file a reviewer can open."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indicquant.agent.runtime import Agent, AgentTurn


def default_sessions_dir() -> Path:
    return Path("data/sessions")


class SessionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, turn: AgentTurn) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")

    def turns(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())


@dataclass
class Session:
    id: str
    agent: Agent
    created_at: str
    title: str = "New chat"
    log: SessionLog | None = None


@dataclass
class SessionHub:
    """In-memory agents plus on-disk traces. One hub per `indicquant serve` process."""

    make_agent: Callable[[], Agent]
    root: Path
    sessions: dict[str, Session] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        agent = self.make_agent()
        log = SessionLog(self.root / f"{sid}.jsonl")
        agent.session_id = sid
        agent.log = log
        sess = Session(
            id=sid,
            agent=agent,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            log=log,
        )
        self.sessions[sid] = sess
        return sess

    def get(self, sid: str) -> Session:
        sess = self.sessions.get(sid)
        if sess is None:
            raise KeyError(sid)
        return sess

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for sess in reversed(list(self.sessions.values())):
            rows.append(
                {
                    "id": sess.id,
                    "created_at": sess.created_at,
                    "title": sess.title,
                    "turns": sess.log.count() if sess.log else 0,
                    "notes": list(sess.agent.memory.notes),
                    "backend": sess.agent.llm.name,
                }
            )
        return rows

    def record(self, sess: Session, user_text: str, turn: AgentTurn) -> None:
        if sess.title == "New chat" and user_text.strip() and user_text != "[confirm]":
            sess.title = user_text.strip()[:48]


def load_trace_files(root: Path | None = None) -> list[dict[str, Any]]:
    """Read persisted sessions from disk (survives a process restart)."""
    root = root or default_sessions_dir()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    paths = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        turns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                turns.append(json.loads(line))
        if not turns:
            continue
        first = turns[0]
        rows.append(
            {
                "id": path.stem,
                "path": str(path),
                "turns": len(turns),
                "last_reply": (turns[-1].get("reply") or "")[:80],
                "language": first.get("language"),
                "grounded": all(t.get("grounded", True) for t in turns),
                "duration_ms": round(sum(float(t.get("duration_ms") or 0) for t in turns), 1),
            }
        )
    return rows
