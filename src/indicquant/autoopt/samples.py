"""Load AutoOpt golden-set formulations (LaTeX/text, not trained weights)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT = Path(__file__).resolve().parents[3] / "evals" / "autoopt.json"


def cases_path() -> Path:
    bundled = Path(__file__).resolve().parents[1] / "eval" / "autoopt.json"
    if bundled.exists():
        return bundled
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "evals" / "autoopt.json"
        if candidate.exists():
            return candidate
    return DEFAULT


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or cases_path()
    rows = json.loads(target.read_text(encoding="utf-8"))
    for row in rows:
        row.setdefault("pipeline", "autoopt")
        row.setdefault("doc_type", "autoopt")
    return rows


def get_case(case_id: str, path: Path | None = None) -> dict[str, Any]:
    for case in load_cases(path):
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)
