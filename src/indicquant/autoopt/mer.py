"""Small MER: grammar-constrained repair on a page transcript.

Not AutoOpt-M1 (393M). Input is Tesseract, paste, or a labelled scan's OCR.
Output is LaTeX/plain math that M2 is allowed to compile.
"""

from __future__ import annotations

import re
from typing import Any

_SENSE = ("minimize", "maximize", "min", "max")
_LE = (("<=", "<="), ("≤", "<="), (r"\leq", "<="), (r"\le", "<="), (" le ", " <= "))
_GE = ((">=", ">="), ("≥", ">="), (r"\geq", ">="), (r"\ge", ">="), (" ge ", " >= "))


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) * len(b) == 0:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = row
    return prev[-1]


def _fix_sense(word: str) -> str:
    low = re.sub(r"[^a-z]", "", word.lower())
    if not low:
        return word
    best = min(_SENSE, key=lambda s: _lev(low, s))
    if _lev(low, best) <= max(2, len(low) // 3):
        return best
    return word


def recognise(transcript: str) -> dict[str, Any]:
    """Map a noisy page string to a compile-ready formulation."""
    raw = transcript.strip()
    if not raw:
        raise ValueError("empty mer transcript")
    text = raw.replace("\r\n", "\n")
    repairs: list[str] = []
    for a, b in (*_LE, *_GE):
        if a in text and a != b:
            text = text.replace(a, b)
            repairs.append(f"{a}->{b}")
    text = re.sub(r"\bs\s*\.\s*t\s*\.?", "s.t.", text, flags=re.I)
    text = re.sub(r"\bs\s+t\b", "s.t.", text, flags=re.I)
    text = re.sub(r"\bsubj(?:ect)?\s+to\b", "subject to", text, flags=re.I)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first, *rest = re.split(r"(\s+)", lines[0], maxsplit=1)
        fixed = _fix_sense(first)
        if fixed != first:
            repairs.append(f"sense:{first}->{fixed}")
            lines[0] = fixed + (rest[0] + rest[1] if rest else "")
    body = "\n".join(lines)
    return {
        "latex": body,
        "source": "mer",
        "chars": len(body),
        "repairs": repairs,
        "raw": raw,
    }
