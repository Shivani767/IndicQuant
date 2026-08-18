"""Jobs: JSON files locally, Postgres+Redis when DATABASE_URL / REDIS_URL are set."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from indicquant.harness.pipeline import run_pipeline
from indicquant.harness.samples import get_case as get_doc_case
from indicquant.harness.samples import load_cases


def default_jobs_dir() -> Path:
    return Path("data/jobs")


class JobBackend(Protocol):
    def get(self, job_id: str) -> dict[str, Any]: ...
    def list(self) -> list[dict[str, Any]]: ...
    def submit(self, doc: dict[str, Any]) -> dict[str, Any]: ...
    def submit_sample(self, sample_id: str) -> dict[str, Any]: ...
    def submit_text(
        self, text: str, doc_type: str | None = None, pipeline: str | None = None
    ) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...


def get_case(sample_id: str) -> dict[str, Any]:
    try:
        return get_doc_case(sample_id)
    except KeyError:
        from indicquant.autoopt.samples import get_case as get_opt_case

        return get_opt_case(sample_id)


def _doc_text(doc: dict[str, Any]) -> str:
    if doc.get("text"):
        return str(doc["text"])
    pages = doc.get("pages") or []
    return "\n".join(str(p.get("text") or "") for p in pages)


def _new_record(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {
            "sample_id": doc.get("id"),
            "pages": len(doc.get("pages") or []),
            "pipeline": doc.get("pipeline"),
        },
    }


def _finish(record: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    from indicquant.autoopt.pipeline import looks_like_opt, run_autoopt

    auto = (
        doc.get("pipeline") == "autoopt"
        or doc.get("doc_type") in {"lp", "autoopt", "optimization"}
        or bool(doc.get("image_b64") or doc.get("image"))
        or looks_like_opt(_doc_text(doc))
    )
    runner = run_autoopt if auto else run_pipeline
    result = runner({**doc, "job_id": record["id"]})
    record.update(
        {
            "status": "done" if result["ok"] else "failed",
            "ok": result["ok"],
            "doc_type": result["doc_type"],
            "result": result["result"],
            "stages": result["stages"],
            "latency_ms": result["latency_ms"],
            "cost": result["cost"],
        }
    )
    return record


class _RedisStatus:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def set_status(self, job_id: str, status: str) -> None:
        pipe = self._r.pipeline()
        pipe.set(f"job:{job_id}:status", status)
        pipe.lpush("jobs:recent", job_id)
        pipe.ltrim("jobs:recent", 0, 49)
        pipe.execute()

    def get_status(self, job_id: str) -> str | None:
        value = self._r.get(f"job:{job_id}:status")
        return str(value) if value else None

    def ping(self) -> bool:
        return bool(self._r.ping())


def _redis_from_env() -> _RedisStatus | None:
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        status = _RedisStatus(url)
        status.ping()
        return status
    except Exception:  # noqa: BLE001 — redis is optional
        return None


class JobStore:
    """JSON files under data/jobs. Used in tests and `uv run` without Docker."""

    def __init__(self, root: Path | None = None, redis: _RedisStatus | None = None) -> None:
        self.root = root or default_jobs_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.redis = redis

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_summary(data))
        return rows[:50]

    def submit(self, doc: dict[str, Any]) -> dict[str, Any]:
        record = _new_record(doc)
        self._write(record)
        if self.redis:
            self.redis.set_status(record["id"], "running")
        record = _finish(record, doc)
        self._write(record)
        if self.redis:
            self.redis.set_status(record["id"], record["status"])
        return record

    def submit_sample(self, sample_id: str) -> dict[str, Any]:
        return self.submit(get_case(sample_id))

    def submit_text(
        self, text: str, doc_type: str | None = None, pipeline: str | None = None
    ) -> dict[str, Any]:
        doc: dict[str, Any] = {"text": text, "doc_type": doc_type}
        if pipeline:
            doc["pipeline"] = pipeline
        return self.submit(doc)

    def health(self) -> dict[str, Any]:
        return {"ok": True, "store": "files", "postgres": False, "redis": self.redis is not None}

    def _write(self, record: dict[str, Any]) -> None:
        self._path(record["id"]).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class PostgresJobStore:
    """Jobs table in Postgres. Index on created_at; full record in jsonb."""

    def __init__(self, url: str, redis: _RedisStatus | None = None) -> None:
        import psycopg

        self.url = url
        self.redis = redis
        self._psycopg = psycopg
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    doc_type TEXT,
                    ok BOOLEAN,
                    latency_ms DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    payload JSONB NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs (created_at DESC)"
            )

    def get(self, job_id: str) -> dict[str, Any]:
        with self._psycopg.connect(self.url) as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row[0]

    def list(self) -> list[dict[str, Any]]:
        with self._psycopg.connect(self.url) as conn:
            rows = conn.execute(
                """
                SELECT id, status, doc_type, ok, latency_ms, created_at
                FROM jobs ORDER BY created_at DESC LIMIT 50
                """
            ).fetchall()
        out = []
        for job_id, status, doc_type, ok, latency_ms, created_at in rows:
            out.append(
                {
                    "id": job_id,
                    "status": status,
                    "doc_type": doc_type,
                    "ok": ok,
                    "latency_ms": latency_ms,
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                }
            )
        return out

    def submit(self, doc: dict[str, Any]) -> dict[str, Any]:
        record = _new_record(doc)
        self._upsert(record)
        if self.redis:
            self.redis.set_status(record["id"], "running")
        record = _finish(record, doc)
        self._upsert(record)
        if self.redis:
            self.redis.set_status(record["id"], record["status"])
        return record

    def submit_sample(self, sample_id: str) -> dict[str, Any]:
        return self.submit(get_case(sample_id))

    def submit_text(
        self, text: str, doc_type: str | None = None, pipeline: str | None = None
    ) -> dict[str, Any]:
        doc: dict[str, Any] = {"text": text, "doc_type": doc_type}
        if pipeline:
            doc["pipeline"] = pipeline
        return self.submit(doc)

    def health(self) -> dict[str, Any]:
        pg = False
        try:
            with self._psycopg.connect(self.url) as conn:
                conn.execute("SELECT 1")
            pg = True
        except Exception:  # noqa: BLE001
            pg = False
        redis_ok = False
        if self.redis:
            try:
                redis_ok = self.redis.ping()
            except Exception:  # noqa: BLE001
                redis_ok = False
        return {"ok": pg, "store": "postgres", "postgres": pg, "redis": redis_ok}

    def _upsert(self, record: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        with self._psycopg.connect(self.url, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, doc_type, ok, latency_ms, created_at, payload)
                VALUES (%s, %s, %s, %s, %s, %s::timestamptz, %s)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    doc_type = EXCLUDED.doc_type,
                    ok = EXCLUDED.ok,
                    latency_ms = EXCLUDED.latency_ms,
                    payload = EXCLUDED.payload
                """,
                (
                    record["id"],
                    record["status"],
                    record.get("doc_type"),
                    record.get("ok"),
                    record.get("latency_ms"),
                    record["created_at"],
                    Jsonb(record),
                ),
            )


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": data["id"],
        "status": data["status"],
        "doc_type": data.get("doc_type"),
        "ok": data.get("ok"),
        "latency_ms": data.get("latency_ms"),
        "created_at": data.get("created_at"),
    }


def open_jobs(root: Path | None = None) -> JobBackend:
    redis = _redis_from_env()
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return PostgresJobStore(url, redis=redis)
    return JobStore(root, redis=redis)


def sample_catalog() -> list[dict[str, Any]]:
    rows = []
    for case in load_cases():
        rows.append(
            {
                "id": case["id"],
                "doc_type": case["doc_type"],
                "language": case.get("language"),
                "pages": len(case.get("pages") or []),
                "pipeline": "documents",
                "preview": (case["pages"][0]["text"][:160] if case.get("pages") else ""),
                "text": "\n\n".join(p["text"] for p in case.get("pages") or []),
            }
        )
    from indicquant.autoopt.samples import load_cases as load_opt

    for case in load_opt():
        body = str(case.get("text") or "")
        rows.append(
            {
                "id": case["id"],
                "doc_type": "autoopt",
                "language": case.get("language") or "en",
                "pages": 1,
                "pipeline": "autoopt",
                "preview": body[:160],
                "text": body,
            }
        )
    return rows
