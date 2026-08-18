"""Pending-action state machine with TTL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from indicquant.agent.pending import PendingAction, PendingStore


def test_create_confirm_cancel(tmp_path) -> None:
    store = PendingStore(tmp_path)
    action = store.create(
        session_id="s1",
        tool="book_ticket",
        args={"origin": "BOM", "destination": "DEL", "passengers": 2},
        preview="2 × BOM → DEL",
        intent={"kind": "book", "language": "hi", "slots": {}, "entities": []},
        assistant_tc=[],
        queued=[],
        messages=[],
    )
    assert action.id.startswith("pnd_")
    assert store.get(action.id).status == "pending"
    store.mark(action.id, "confirmed")
    assert store.get(action.id).status == "confirmed"


def test_expired_cannot_confirm(tmp_path) -> None:
    store = PendingStore(tmp_path)
    action = PendingAction(
        id="pnd_old",
        session_id="s1",
        tool="book_ticket",
        args={},
        preview="x",
        created_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        expires_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        intent={},
    )
    store.save(action)
    got = store.get("pnd_old")
    assert got.status == "expired"
    try:
        store.mark("pnd_old", "confirmed")
        raise AssertionError("expired action should not confirm")
    except ValueError:
        pass
