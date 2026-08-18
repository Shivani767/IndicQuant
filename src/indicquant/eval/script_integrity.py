"""Script integrity — the failure mode accuracy metrics cannot see.

The spec's argument, made concrete: MCQ accuracy is a poor proxy for what quantization
actually breaks. A model that answers a Hindi question correctly but emits it in Latin
script, or with broken Devanagari conjuncts, has failed in a way no accuracy number reports.

Three measurements, all computable without a GPU:

  1. SCRIPT PURITY   — fraction of output characters in the expected Unicode block
  2. SCRIPT DRIFT    — output that silently switches to English/Latin mid-generation
  3. MALFORMED GRAPHEMES — orphaned combining marks, dangling viramas, stray ZWJ/ZWNJ.
     Indic scripts encode conjuncts as base + virara + base sequences, so quantization
     damage tends to show up as broken cluster structure before it shows up as wrong words.
     This is the most Indic-specific signal in the project and has no English analogue.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

# Unicode ranges per script. Used instead of character-name matching because it is exact and
# cheap; the name-based path in systems/fertility.py is the lenient variant used for corpus
# filtering, where a false negative merely drops a document.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin": [(0x0020, 0x024F)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF), (0x11FC0, 0x11FFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
}

# Virama (halant) per script — the conjunct-forming character. A virama at the end of a word
# with nothing following it is a broken cluster.
VIRAMA = {
    "Devanagari": "्",
    "Bengali": "্",
    "Gurmukhi": "੍",
    "Gujarati": "્",
    "Odia": "୍",
    "Tamil": "்",
    "Telugu": "్",
    "Kannada": "್",
    "Malayalam": "്",
}

ZWJ = "‍"
ZWNJ = "‌"

_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


@dataclass
class ScriptIntegrityResult:
    language: str
    script: str
    n_samples: int
    script_purity: float
    """Mean fraction of non-space, non-punctuation characters in the expected script."""

    drift_rate: float
    """Fraction of outputs that contain a run of Latin words while a non-Latin script was
    expected — the model silently answering in English."""

    malformed_rate: float
    """Fraction of outputs containing at least one broken grapheme cluster."""

    orphan_combining_marks: int
    dangling_virama: int
    stray_zwj: int
    empty_outputs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def in_script(char: str, script: str) -> bool:
    ranges = SCRIPT_RANGES.get(script)
    if not ranges:
        return False
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_purity(text: str, script: str) -> float:
    """Fraction of meaningful characters in the expected script.

    Digits, punctuation and whitespace are excluded: they are script-neutral in practice
    (Indic text routinely uses ASCII digits and punctuation), and counting them would
    inflate purity for every language equally while hiding real drift.
    """
    chars = [
        c
        for c in text
        if not c.isspace() and not c.isdigit() and unicodedata.category(c)[0] not in {"P", "S"}
    ]
    if not chars:
        return 0.0
    return sum(1 for c in chars if in_script(c, script)) / len(chars)


def has_script_drift(text: str, script: str, min_latin_words: int = 3) -> bool:
    """Whether a non-Latin-script response drifted into English."""
    if script == "Latin":
        return False
    return len(_LATIN_WORD.findall(text)) >= min_latin_words


def count_malformed(text: str, script: str) -> dict[str, int]:
    """Count broken grapheme structures.

    - orphan combining mark: a combining mark with no base character before it
    - dangling virama: a virama at the end of a token with no consonant following, i.e. a
      conjunct that was started and never completed
    - stray ZWJ/ZWNJ: a joiner at a string boundary, where it can join nothing
    """
    virama = VIRAMA.get(script)
    orphans = 0
    dangling = 0
    stray = 0

    for i, ch in enumerate(text):
        # Category, not unicodedata.combining(). Most Indic vowel signs are SPACING
        # combining marks (category Mc) with canonical combining class 0, so
        # combining() returns 0 for them and they would never be counted — silently
        # under-reporting the most common malformation in Devanagari, Tamil and Bengali.
        if unicodedata.category(ch) in {"Mn", "Mc", "Me"}:
            if i == 0 or text[i - 1].isspace():
                orphans += 1
        if virama and ch == virama:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == "" or nxt.isspace():
                dangling += 1
        if ch in (ZWJ, ZWNJ):
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not prev or not nxt or prev.isspace() or nxt.isspace():
                stray += 1

    return {
        "orphan_combining_marks": orphans,
        "dangling_virama": dangling,
        "stray_zwj": stray,
    }


def measure_script_integrity(
    outputs: list[str], language: str, script: str
) -> ScriptIntegrityResult:
    """Measure script integrity over a set of generations for one language."""
    if not outputs:
        raise ValueError("no outputs to measure")

    purities = []
    drifted = 0
    malformed = 0
    empty = 0
    totals = {"orphan_combining_marks": 0, "dangling_virama": 0, "stray_zwj": 0}

    for text in outputs:
        if not text.strip():
            empty += 1
            purities.append(0.0)
            continue
        purities.append(script_purity(text, script))
        if has_script_drift(text, script):
            drifted += 1
        counts = count_malformed(text, script)
        for key, value in counts.items():
            totals[key] += value
        if any(counts.values()):
            malformed += 1

    n = len(outputs)
    return ScriptIntegrityResult(
        language=language,
        script=script,
        n_samples=n,
        script_purity=sum(purities) / n,
        drift_rate=drifted / n,
        malformed_rate=malformed / n,
        orphan_combining_marks=totals["orphan_combining_marks"],
        dangling_virama=totals["dangling_virama"],
        stray_zwj=totals["stray_zwj"],
        empty_outputs=empty,
    )
