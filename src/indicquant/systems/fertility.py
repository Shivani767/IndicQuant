"""Tokenizer fertility per language.

THE PHASE A RESULT. This needs `tokenizer.json` (a few MB) and nothing else — no GPU, no
128 GB download. It produces a real, publishable measurement on day one.

Why it matters beyond being cheap: Indic scripts yield more tokens per word than English, so
an equal-length prompt costs more tokens, occupies more KV cache, and takes longer to
prefill. That inflates every downstream latency, memory and cost number, and it compounds
with everything else the project measures. It is also a potential confound for RQ3 — if
Odia degrades more than Hindi under quantization, part of that could be fertility rather than
calibration mismatch, so it must be measured before it can be controlled for.

METHOD: fertility is measured on FLORES-200 parallel text. Every language contains the SAME
content, translated, so tokens-per-word differences are attributable to the tokenizer rather
than to what happens to be in each corpus. Measuring fertility on non-parallel corpora is the
standard way to get a confounded number.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from indicquant.config import Language, load_languages

# Indic scripts do not use spaces the way Latin does for all boundaries, but word-level
# segmentation by whitespace is the convention in the fertility literature and keeps the
# comparison honest across scripts. We report characters-per-token alongside it precisely
# because whitespace segmentation flatters some scripts over others.
_WORD_RE = re.compile(r"\S+")


@dataclass
class FertilityResult:
    language: str
    language_name: str
    script: str
    tier: str
    resource_rank: int
    n_sentences: int
    n_words: int
    n_chars: int
    n_tokens: int
    tokens_per_word: float
    tokens_per_char: float
    chars_per_token: float
    fertility_ratio_vs_english: float | None = None
    """tokens_per_word relative to English on the same parallel content. This is the number
    that says 'an Odia prompt costs Nx more tokens than the identical English prompt'."""

    unk_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_fertility(
    texts_by_language: dict[str, list[str]],
    tokenizer: Any,
    languages: Any = None,
    reference_language: str = "en",
) -> list[FertilityResult]:
    """Measure fertility across languages on parallel text.

    `texts_by_language` maps language code to a list of sentences. For a valid comparison
    each list must be translations of the same source sentences, in the same order.
    """
    langset = languages if languages is not None else load_languages()
    results: list[FertilityResult] = []

    lengths = {code: len(texts) for code, texts in texts_by_language.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"parallel corpus required, got differing sentence counts: {lengths}. "
            "Fertility measured on non-parallel text confounds the tokenizer with corpus "
            "content."
        )

    for code, texts in texts_by_language.items():
        lang: Language = langset.by_code(code)
        n_words = sum(len(_WORD_RE.findall(t)) for t in texts)
        n_chars = sum(len(t) for t in texts)
        encoded = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        n_tokens = sum(len(ids) for ids in encoded)

        unk_id = getattr(tokenizer, "unk_token_id", None)
        unk_count = (
            sum(sum(1 for i in ids if i == unk_id) for ids in encoded) if unk_id is not None else 0
        )

        results.append(
            FertilityResult(
                language=code,
                language_name=lang.name,
                script=lang.script,
                tier=lang.tier,
                resource_rank=lang.resource_rank,
                n_sentences=len(texts),
                n_words=n_words,
                n_chars=n_chars,
                n_tokens=n_tokens,
                tokens_per_word=n_tokens / n_words if n_words else float("nan"),
                tokens_per_char=n_tokens / n_chars if n_chars else float("nan"),
                chars_per_token=n_chars / n_tokens if n_tokens else float("nan"),
                unk_rate=unk_count / n_tokens if n_tokens else 0.0,
            )
        )

    ref = next((r for r in results if r.language == reference_language), None)
    if ref is not None and ref.tokens_per_word > 0:
        for r in results:
            r.fertility_ratio_vs_english = r.tokens_per_word / ref.tokens_per_word

    results.sort(key=lambda r: r.resource_rank)
    return results


def script_share(text: str, expected_blocks: tuple[str, ...]) -> float:
    """Fraction of a string's non-space characters in the expected Unicode block(s).

    Shared with `eval/script_integrity.py` and used by the corpus builders to reject
    documents that are nominally Indic but mostly English boilerplate — a real hazard in
    web-crawled corpora, and one that would silently turn the "Indic" calibration set into a
    mixed one, destroying the B-vs-E comparison.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    wanted = {b.upper().replace("_", " ") for b in expected_blocks}
    hits = 0
    for c in chars:
        try:
            name = unicodedata.name(c)
        except ValueError:
            continue
        if any(name.startswith(w.split(" ")[0]) for w in wanted):
            hits += 1
        elif "BASIC LATIN" in wanted and ord(c) < 128:
            hits += 1
    return hits / len(chars)


def write_results(results: list[FertilityResult], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    return out_path


def format_table(results: list[FertilityResult]) -> str:
    """Markdown table, for pasting straight into the blog post."""
    header = (
        "| Language | Script | Tier | Tokens/word | Chars/token | vs. English |\n"
        "|---|---|---|---:|---:|---:|"
    )
    rows = [
        f"| {r.language_name} | {r.script} | {r.tier} | {r.tokens_per_word:.2f} | "
        f"{r.chars_per_token:.2f} | "
        f"{r.fertility_ratio_vs_english:.2f}x |"
        if r.fertility_ratio_vs_english is not None
        else f"| {r.language_name} | {r.script} | {r.tier} | {r.tokens_per_word:.2f} | "
        f"{r.chars_per_token:.2f} | — |"
        for r in results
    ]
    return "\n".join([header, *rows])
