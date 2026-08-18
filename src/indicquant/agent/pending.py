"""Pending side-effect actions. Confirm/cancel is a state machine with TTL, not a prompt."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

Status = Literal["pending", "confirmed", "cancelled", "expired"]

DEFAULT_TTL_SECONDS = 15 * 60


def default_pending_dir() -> Path:
    return Path("data/pending")


def _now() -> datetime:
    return datetime.now(UTC)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass
class PendingAction:
    id: str
    session_id: str
    tool: str
    args: dict[str, Any]
    preview: str
    created_at: str
    expires_at: str
    status: Status = "pending"
    intent: dict[str, Any] = field(default_factory=dict)
    assistant_tc: list[dict[str, Any]] = field(default_factory=list)
    queued: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.status != "pending":
            return self.status == "expired"
        return (now or _now()) >= _parse(self.expires_at)

    def refresh_status(self) -> Status:
        if self.status == "pending" and self.is_expired():
            self.status = "expired"
        return self.status


class PendingStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_pending_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, action_id: str) -> Path:
        return self.root / f"{action_id}.json"

    def create(
        self,
        *,
        session_id: str,
        tool: str,
        args: dict[str, Any],
        preview: str,
        intent: dict[str, Any],
        assistant_tc: list[dict[str, Any]],
        queued: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        raw: dict[str, Any] | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> PendingAction:
        created = _now()
        action = PendingAction(
            id=f"pnd_{uuid.uuid4().hex[:10]}",
            session_id=session_id,
            tool=tool,
            args=args,
            preview=preview,
            created_at=created.isoformat(timespec="seconds"),
            expires_at=(created + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds"),
            intent=intent,
            assistant_tc=assistant_tc,
            queued=queued,
            messages=messages,
            raw=raw or {},
        )
        self.save(action)
        return action

    def save(self, action: PendingAction) -> None:
        self._path(action.id).write_text(
            json.dumps(action.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, action_id: str) -> PendingAction:
        path = self._path(action_id)
        if not path.exists():
            raise KeyError(action_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        action = PendingAction(**data)
        if action.refresh_status() == "expired":
            self.save(action)
        return action

    def list(self, *, include_closed: bool = False) -> list[PendingAction]:
        rows: list[PendingAction] = []
        for path in sorted(self.root.glob("pnd_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                action = self.get(path.stem)
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if include_closed or action.status == "pending":
                rows.append(action)
        return rows

    def mark(self, action_id: str, status: Status) -> PendingAction:
        action = self.get(action_id)
        if action.status != "pending":
            raise ValueError(f"{action_id} is {action.status}, not pending")
        if status == "confirmed" and action.is_expired():
            action.status = "expired"
            self.save(action)
            raise ValueError(f"{action_id} expired at {action.expires_at}")
        action.status = status
        self.save(action)
        return action
