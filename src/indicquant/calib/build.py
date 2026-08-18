"""Calibration corpus assembly.

THE INVARIANT: corpora are filled to an exact TOKEN budget, not a document count.

Conditions B, E, F, H and I differ only in calibration distribution. If the English corpus
carried 1.0M tokens and the Indic corpus 1.4M — easy to do accidentally, since Indic
tokenizer fertility is higher, so equal document counts give unequal token counts — then any
difference in outcome would confound distribution with volume, and the project's headline
result would be uninterpretable.

`verify_budget_parity()` enforces this and `tests/test_calib_budget.py` asserts it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indicquant.config import Manifest, load_calibration_config, load_languages


class CalibrationBudgetError(RuntimeError):
    """Raised when corpora that must be volume-matched are not."""


@dataclass
class CalibrationCorpus:
    """A built calibration corpus plus the provenance needed to defend it."""

    name: str
    sequences: list[str]
    n_tokens: int
    seq_len: int
    seed: int
    language_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    holdout_languages: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sequences)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_sequences": len(self.sequences),
            "n_tokens": self.n_tokens,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "language_counts": self.language_counts,
            "source_counts": self.source_counts,
            "holdout_languages": self.holdout_languages,
        }

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / "sequences.jsonl"
        with data_path.open("w") as f:
            for i, seq in enumerate(self.sequences):
                f.write(json.dumps({"id": i, "text": seq}) + "\n")
        Manifest(
            stage="calibration",
            config=self.summary(),
            outputs=[str(data_path)],
            metrics={"n_tokens": self.n_tokens, "n_sequences": len(self.sequences)},
        ).write(out_dir / "manifest.json")
        return data_path

    @classmethod
    def load(cls, out_dir: str | Path) -> CalibrationCorpus:
        out_dir = Path(out_dir)
        with (out_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        cfg = manifest["config"]
        sequences = []
        with (out_dir / "sequences.jsonl").open() as f:
            for line in f:
                sequences.append(json.loads(line)["text"])
        return cls(
            name=cfg["name"],
            sequences=sequences,
            n_tokens=cfg["n_tokens"],
            seq_len=cfg["seq_len"],
            seed=cfg["seed"],
            language_counts=cfg.get("language_counts", {}),
            source_counts=cfg.get("source_counts", {}),
            holdout_languages=cfg.get("holdout_languages", []),
        )


def build_corpus(
    config_name: str,
    tokenizer: Any,
    documents_by_source: dict[str, Any] | None = None,
) -> CalibrationCorpus:
    """Build a calibration corpus to its configured token budget.

    `documents_by_source` maps a source key to an iterable of `{"text": ..., "language": ...}`
    dicts. Passing it explicitly keeps this function testable offline; when omitted the
    sources named in the config are streamed from Hugging Face.
    """
    cfg = load_calibration_config(config_name)
    budget = cfg["token_budget"]
    seq_len = cfg["seq_len"]
    seed = cfg.get("seed", 0)
    rng = random.Random(seed)

    if documents_by_source is None:
        from indicquant.calib.corpora import stream_sources

        documents_by_source = stream_sources(cfg)

    filters = cfg.get("filters", {})
    langset = load_languages()

    sequences: list[str] = []
    total_tokens = 0
    language_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    seen_hashes: set[int] = set()

    # Round-robin across sources weighted by config, so a single large source cannot
    # dominate the budget. Interleaving rather than concatenating also means truncating at
    # the budget does not silently drop the last source entirely.
    pools = _weighted_interleave(documents_by_source, cfg, rng)

    for source_key, doc in pools:
        if total_tokens >= budget:
            break
        text = doc.get("text", "")
        if not _passes_filters(text, doc, filters, langset):
            continue
        if filters.get("dedup") == "exact":
            h = hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(ids) - seq_len + 1, seq_len):
            if total_tokens >= budget:
                break
            chunk = ids[start : start + seq_len]
            sequences.append(tokenizer.decode(chunk))
            total_tokens += len(chunk)
            lang = doc.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + len(chunk)
            source_counts[source_key] = source_counts.get(source_key, 0) + len(chunk)

    if total_tokens < budget:
        raise CalibrationBudgetError(
            f"corpus {config_name!r} exhausted its sources at {total_tokens:,} tokens, short "
            f"of the {budget:,} budget. Every corpus must reach the same budget or the "
            "distribution comparison is confounded with volume. Widen the sources or lower "
            "the budget for ALL corpora together."
        )

    return CalibrationCorpus(
        name=cfg["name"],
        sequences=sequences,
        n_tokens=total_tokens,
        seq_len=seq_len,
        seed=seed,
        language_counts=language_counts,
        source_counts=source_counts,
        holdout_languages=list(cfg.get("holdout_languages", [])),
    )


def _weighted_interleave(
    documents_by_source: dict[str, Any], cfg: dict[str, Any], rng: random.Random
):
    """Yield (source_key, document) pairs, sampling sources by configured weight."""
    weights = {}
    for source in cfg.get("sources", []):
        key = source.get("hf_id") or source.get("local") or "unknown"
        weights[key] = source.get("weight", 1.0)
    if not weights:
        weights = dict.fromkeys(documents_by_source, 1.0)

    iterators = {k: iter(v) for k, v in documents_by_source.items() if k in weights}
    keys = list(iterators)
    if not keys:
        return

    while iterators:
        keys = list(iterators)
        pick = rng.choices(keys, weights=[weights.get(k, 1.0) for k in keys], k=1)[0]
        try:
            yield pick, next(iterators[pick])
        except StopIteration:
            del iterators[pick]


def _passes_filters(text: str, doc: dict[str, Any], filters: dict[str, Any], langset: Any) -> bool:
    if len(text) < filters.get("min_chars", 0):
        return False
    if len(text) > filters.get("max_chars", float("inf")):
        return False

    purity = filters.get("script_purity")
    if purity:
        # Web-crawled Indic corpora carry English boilerplate. Without this check the
        # "Indic" calibration set is quietly a mixed one, and B-vs-E stops meaning what it
        # claims to mean.
        from indicquant.systems.fertility import script_share

        lang_code = doc.get("language")
        if lang_code:
            try:
                lang = langset.by_code(lang_code)
            except Exception:
                return True
            if lang.unicode_blocks and script_share(text, lang.unicode_blocks) < purity:
                return False
    return True


def verify_budget_parity(corpora: list[CalibrationCorpus], tolerance: float = 0.01) -> None:
    """Assert that volume-matched corpora really are volume-matched.

    Called before any B/E/F comparison is reported. Raises rather than warns: a silent
    mismatch here invalidates the project's headline result, and a warning in a log is not
    a control.
    """
    if len(corpora) < 2:
        return
    counts = {c.name: c.n_tokens for c in corpora}
    lo, hi = min(counts.values()), max(counts.values())
    if lo <= 0:
        raise CalibrationBudgetError(f"corpus with zero tokens: {counts}")
    if (hi - lo) / lo > tolerance:
        raise CalibrationBudgetError(
            f"calibration corpora are not volume-matched (spread {(hi - lo) / lo:.1%} > "
            f"{tolerance:.1%}): {counts}. Conditions B/E/F must differ in DISTRIBUTION only "
            "— see ARCHITECTURE.md §3.1."
        )
