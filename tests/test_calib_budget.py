"""Calibration budget parity — the invariant that makes B vs. E interpretable.

If the English corpus carried 1.0M tokens and the Indic corpus 1.4M, any measured difference
would confound calibration DISTRIBUTION with calibration VOLUME, and the project's headline
result would mean nothing. This is easy to get wrong by accident: Indic tokenizer fertility
is higher, so equal document counts give unequal token counts.
"""

from __future__ import annotations

import pytest

from indicquant.calib.build import (
    CalibrationBudgetError,
    CalibrationCorpus,
    verify_budget_parity,
)
from indicquant.config import load_calibration_config

VOLUME_MATCHED = ["english_c4", "indic_sangraha", "mixed", "codemixed"]


def _corpus(name: str, n_tokens: int) -> CalibrationCorpus:
    return CalibrationCorpus(name=name, sequences=[], n_tokens=n_tokens, seq_len=2048, seed=0)


@pytest.mark.parametrize("name", VOLUME_MATCHED)
def test_config_declares_a_token_budget(name):
    cfg = load_calibration_config(name)
    assert cfg["token_budget"] > 0
    assert cfg["seq_len"] > 0


def test_all_configs_share_one_budget():
    """The configs themselves must agree before any corpus is built."""
    budgets = {n: load_calibration_config(n)["token_budget"] for n in VOLUME_MATCHED}
    assert len(set(budgets.values())) == 1, f"calibration budgets diverge: {budgets}"

    seq_lens = {n: load_calibration_config(n)["seq_len"] for n in VOLUME_MATCHED}
    assert len(set(seq_lens.values())) == 1, f"sequence lengths diverge: {seq_lens}"


def test_matched_corpora_pass():
    verify_budget_parity([_corpus("a", 1_048_576), _corpus("b", 1_048_576)])


def test_slight_mismatch_within_tolerance_passes():
    verify_budget_parity([_corpus("a", 1_000_000), _corpus("b", 1_005_000)], tolerance=0.01)


def test_mismatched_corpora_raise():
    with pytest.raises(CalibrationBudgetError, match="not volume-matched"):
        verify_budget_parity([_corpus("english", 1_000_000), _corpus("indic", 1_400_000)])


def test_empty_corpus_raises():
    with pytest.raises(CalibrationBudgetError):
        verify_budget_parity([_corpus("a", 0), _corpus("b", 1_000_000)])


def test_single_corpus_is_trivially_fine():
    verify_budget_parity([_corpus("a", 1_000)])


def test_indic_corpus_holds_out_a_language():
    """RQ5 needs a language that was never calibrated on, or transfer is untestable."""
    cfg = load_calibration_config("indic_sangraha")
    holdout = cfg.get("holdout_languages", [])
    assert holdout, "no holdout language — RQ5 (cross-language transfer) becomes unanswerable"
    calibrated = cfg["sources"][0]["languages"]
    assert not set(holdout) & set(calibrated)


def test_corpus_roundtrips_through_disk(tmp_path):
    corpus = CalibrationCorpus(
        name="test",
        sequences=["एक दो तीन", "hello world"],
        n_tokens=10,
        seq_len=2048,
        seed=0,
        language_counts={"hi": 5, "en": 5},
    )
    corpus.save(tmp_path)
    loaded = CalibrationCorpus.load(tmp_path)
    assert loaded.sequences == corpus.sequences
    assert loaded.n_tokens == corpus.n_tokens
    assert loaded.language_counts == corpus.language_counts
