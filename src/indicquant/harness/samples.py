"""Load golden-set documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT = Path(__file__).resolve().parents[3] / "evals" / "documents.json"


def cases_path() -> Path:
    bundled = Path(__file__).resolve().parents[1] / "eval" / "documents.json"
    if bundled.exists():
        return bundled
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "evals" / "documents.json"
        if candidate.exists():
            return candidate
    return DEFAULT


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or cases_path()
    return json.loads(target.read_text(encoding="utf-8"))


def get_case(case_id: str, path: Path | None = None) -> dict[str, Any]:
    for case in load_cases(path):
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)
