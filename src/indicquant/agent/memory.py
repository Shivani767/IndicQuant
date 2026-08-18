"""Cross-turn memory with provenance. Inferred facts are not reused as truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Source = Literal["tool_call", "user_stated", "inferred"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class MemoryRecord:
    fact: str
    source: Source
    timestamp: str
    verified: bool
    tool: str | None = None

    @property
    def usable(self) -> bool:
        if self.source == "inferred":
            return self.verified
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodicStore:
    records: list[MemoryRecord] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return [r.fact for r in self.records]

    def snapshot(self) -> list[str]:
        return list(self.notes)

    def remember(
        self,
        note: str,
        source: Source = "user_stated",
        *,
        verified: bool = False,
        tool: str | None = None,
    ) -> MemoryRecord:
        fact = note.strip()
        rec = MemoryRecord(
            fact=fact,
            source=source,
            timestamp=_now(),
            verified=verified,
            tool=tool,
        )
        if fact:
            existing = next((r for r in self.records if r.fact == fact), None)
            if existing is None:
                self.records.append(rec)
            else:
                existing.source = source
                existing.verified = verified or existing.verified
                existing.tool = tool or existing.tool
                existing.timestamp = rec.timestamp
                rec = existing
        return rec

    def recall_records(self, query: str = "") -> list[MemoryRecord]:
        q = query.strip().casefold()
        rows = list(self.records)
        if q:
            rows = [r for r in rows if q in r.fact.casefold()]
        return rows

    def recall(self, query: str = "") -> list[str]:
        """Facts safe to reuse. Inferred-and-unverified rows are dropped."""
        return [r.fact for r in self.recall_records(query) if r.usable]

    def diff(self, before: list[str]) -> dict[str, list[str]]:
        now = set(self.notes)
        old = set(before)
        return {
            "added": [n for n in self.notes if n not in old],
            "removed": [n for n in before if n not in now],
        }
