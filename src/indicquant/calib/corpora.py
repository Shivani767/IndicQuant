"""Calibration data sources.

All source IDs were checked on Hugging Face on 2026-08-18. Re-verify before Phase 0 —
dataset configs and split names drift.

  English    : allenai/c4                (the near-universal default; the thing we test)
  Indic      : ai4bharat/sangraha        (verified split), ai4bharat/IndicCorpV2
  Code-mixed : constructed — see build_codemixed_corpus()
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

SOURCES = {
    "allenai/c4": {
        "config": "en",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "note": "Standard English calibration corpus. Conditions B, C, D, G.",
    },
    "ai4bharat/sangraha": {
        "config": "verified",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "note": "Largest curated Indic corpus. The 'verified' split is human-curated; "
        "'synthetic' and 'romanised' splits exist and the romanised one is worth "
        "checking as a natural code-mixed source before falling back to "
        "transliteration.",
    },
    "ai4bharat/IndicCorpV2": {
        "config": None,
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "note": "Fallback / supplement to sangraha.",
    },
}


def stream_sources(config: dict[str, Any]) -> dict[str, Iterator[dict[str, Any]]]:
    """Open a streaming iterator per configured source.

    Streaming, not downloading: the calibration budget is ~1M tokens, a rounding error
    against C4's full size, and there is no reason to pull terabytes to sample a megabyte.
    """
    from datasets import load_dataset

    streams: dict[str, Iterator[dict[str, Any]]] = {}
    for source in config.get("sources", []):
        hf_id = source.get("hf_id")
        if hf_id is None:
            continue  # local sources are read by the caller
        languages = source.get("languages") or [source.get("language")]
        holdout = set(config.get("holdout_languages", []))
        languages = [lang for lang in languages if lang and lang not in holdout]

        streams[hf_id] = _multilingual_stream(
            load_dataset_fn=load_dataset,
            hf_id=hf_id,
            base_config=source.get("config"),
            split=source.get("split", "train"),
            text_field=source.get("text_field", "text"),
            languages=languages,
            weighting=source.get("language_weighting", "uniform"),
        )
    return streams


def _multilingual_stream(
    load_dataset_fn: Any,
    hf_id: str,
    base_config: str | None,
    split: str,
    text_field: str,
    languages: list[str],
    weighting: str,
) -> Iterator[dict[str, Any]]:
    """Interleave per-language streams, tagging each document with its language.

    Uniform language weighting is deliberate for the Indic corpus: weighting by pretraining
    share would reproduce the very imbalance the project argues against. See
    configs/calibration/indic_sangraha.yaml.
    """
    import itertools

    per_lang = {}
    for lang in languages:
        try:
            cfg = f"{base_config}-{lang}" if base_config and lang else base_config
            ds = load_dataset_fn(hf_id, cfg, split=split, streaming=True)
            per_lang[lang] = iter(ds)
        except Exception:  # noqa: BLE001 - config naming varies per dataset
            continue

    if not per_lang:
        ds = load_dataset_fn(hf_id, base_config, split=split, streaming=True)
        for doc in ds:
            yield {"text": doc.get(text_field, ""), "language": languages[0] if languages else None}
        return

    if weighting != "uniform":
        raise ValueError(f"unsupported language_weighting: {weighting!r}")

    # Round-robin gives uniform language representation regardless of source size.
    for lang in itertools.cycle(list(per_lang)):
        if not per_lang:
            return
        try:
            doc = next(per_lang[lang])
        except StopIteration:
            per_lang.pop(lang, None)
            continue
        except KeyError:
            continue
        yield {"text": doc.get(text_field, ""), "language": lang}


def build_codemixed_corpus(
    config: dict[str, Any],
    transliterator: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Construct the romanized Hinglish/Tanglish corpus.

    No canonical corpus of the needed scale exists — checked HF on 2026-08-18:
    cmu_hinglish_dog, LinCE and similar are small and mostly classification-oriented.
    Building this is a contribution in itself, as the spec anticipated.

    Two blended routes, with the ratio recorded in the manifest:
      - natural (0.4)        : real romanized text where it exists
      - transliterated (0.6) : native-script Indic text romanized via ai4bharat/Aksharantar,
                               with English tokens injected to approximate code-switching

    The caveat travels with the data: transliterated text is cleaner than organic typing —
    consistent romanization, no spelling variation — so it understates the real difficulty
    of H4. Any result computed on this corpus reports the natural/transliterated ratio
    alongside it.
    """
    raise NotImplementedError(
        "Phase A deliverable, and a contribution in its own right.\n"
        "Steps:\n"
        "  1. Check ai4bharat/sangraha's 'romanised' split first — if it is large enough, "
        "     the natural route may cover the whole budget and the caveat disappears.\n"
        "  2. Otherwise: pull hi/ta from sangraha 'verified', romanize word-by-word via "
        "     ai4bharat/Aksharantar, inject English at english_injection_rate.\n"
        "  3. Record the natural/transliterated ratio in the manifest.\n"
        "  4. Hold out a slice as the code-mixed EVAL set — it must not overlap "
        "     calibration."
    )
