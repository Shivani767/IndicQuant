"""Script integrity tests.

These measure the failure mode accuracy metrics cannot see: a model that answers correctly in
the wrong script, or emits broken Devanagari conjuncts.
"""

from __future__ import annotations

from indicquant.eval.script_integrity import (
    count_malformed,
    has_script_drift,
    measure_script_integrity,
    script_purity,
)


def test_pure_devanagari_scores_one():
    assert script_purity("यह हिंदी वाक्य है", "Devanagari") == 1.0


def test_pure_tamil_scores_one():
    assert script_purity("இது தமிழ் வாக்கியம்", "Tamil") == 1.0


def test_wrong_script_scores_zero():
    assert script_purity("this is english", "Devanagari") == 0.0


def test_mixed_script_scores_between():
    purity = script_purity("यह hindi है", "Devanagari")
    assert 0.0 < purity < 1.0


def test_punctuation_and_digits_do_not_count():
    """Indic text routinely uses ASCII digits and punctuation. Counting them would depress
    purity uniformly and mask real drift."""
    assert script_purity("यह हिंदी है, 123!", "Devanagari") == 1.0


def test_empty_string_scores_zero():
    assert script_purity("", "Devanagari") == 0.0
    assert script_purity("   ", "Devanagari") == 0.0


def test_drift_detected_when_model_answers_in_english():
    assert has_script_drift("the answer is clearly this", "Devanagari")


def test_no_drift_for_pure_indic():
    assert not has_script_drift("यह हिंदी वाक्य है", "Devanagari")


def test_a_stray_english_word_is_not_drift():
    """Code-switching a single word is normal Indic usage, not a failure."""
    assert not has_script_drift("यह एक computer है", "Devanagari")


def test_latin_script_languages_never_drift():
    """Hinglish is romanized by definition, so Latin output is correct there."""
    assert not has_script_drift("yeh sahi hai bhai", "Latin")


def test_dangling_virama_detected():
    """A virama with nothing after it is a conjunct that was started and never finished —
    a characteristic quantization artifact with no English analogue."""
    counts = count_malformed("क्", "Devanagari")
    assert counts["dangling_virama"] == 1


def test_well_formed_conjunct_is_clean():
    counts = count_malformed("क्ष", "Devanagari")
    assert counts["dangling_virama"] == 0
    assert counts["orphan_combining_marks"] == 0


def test_orphan_combining_mark_detected():
    counts = count_malformed(" ि", "Devanagari")
    assert counts["orphan_combining_marks"] == 1


def test_measure_aggregates_over_samples():
    outputs = [
        "यह हिंदी वाक्य है",  # clean
        "this is english text",  # drift
        "क्",  # malformed
        "",  # empty
    ]
    result = measure_script_integrity(outputs, language="hi", script="Devanagari")
    assert result.n_samples == 4
    assert result.empty_outputs == 1
    assert result.drift_rate == 0.25
    assert result.malformed_rate == 0.25
    assert 0.0 < result.script_purity < 1.0


def test_all_clean_outputs_score_perfectly():
    result = measure_script_integrity(
        ["यह हिंदी है", "वह भी हिंदी है"], language="hi", script="Devanagari"
    )
    assert result.script_purity == 1.0
    assert result.drift_rate == 0.0
    assert result.malformed_rate == 0.0
