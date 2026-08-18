"""Statistics tests."""

from __future__ import annotations

import numpy as np
import pytest

from indicquant.analysis.stats import bootstrap_ci, paired_delta, resource_trend


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=50)
    ci = bootstrap_ci(values, seed=0)
    assert ci.lo < ci.mean < ci.hi
    assert ci.mean == pytest.approx(values.mean())


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.normal(0, 1, size=10), seed=0)
    large = bootstrap_ci(rng.normal(0, 1, size=1000), seed=0)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_bootstrap_ci_on_a_single_value_is_degenerate():
    ci = bootstrap_ci([0.42])
    assert ci.mean == ci.lo == ci.hi == 0.42


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_excludes_zero():
    assert bootstrap_ci([1.0, 1.1, 0.9, 1.05], seed=0).excludes_zero()
    assert not bootstrap_ci([-1.0, 1.0, -0.5, 0.5], seed=0).excludes_zero()


def test_paired_delta_detects_a_real_effect():
    """The B-vs-E shape: paired per-example scores with a consistent shift."""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.6, 0.05, size=200)
    treatment = baseline + 0.04
    result = paired_delta(baseline, treatment, seed=0)
    assert result.delta.mean == pytest.approx(0.04, abs=1e-6)
    assert result.significant
    assert result.p_value < 0.01


def test_paired_delta_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.6, 0.05, size=200)
    treatment = rng.normal(0.6, 0.05, size=200)
    assert not paired_delta(baseline, treatment, seed=0).significant


def test_paired_delta_requires_aligned_arrays():
    """Conditions must be evaluated on identical examples for the pairing to be real."""
    with pytest.raises(ValueError, match="aligned"):
        paired_delta([1, 2, 3], [1, 2])


def test_resource_trend_recovers_a_planted_slope():
    ranks = list(range(12))
    deltas = [-0.01 * r for r in ranks]
    trend = resource_trend(ranks, deltas, n_resamples=500, seed=0)
    assert trend.slope == pytest.approx(-0.01, abs=1e-9)
    assert trend.r_squared > 0.99
    assert trend.supports_h1


def test_resource_trend_rejects_a_flat_relationship():
    """The null result: quantization hurts every language equally. H1 must not fire."""
    rng = np.random.default_rng(0)
    ranks = list(range(12))
    deltas = list(rng.normal(-0.05, 0.005, size=12))
    trend = resource_trend(ranks, deltas, n_resamples=500, seed=0)
    assert not trend.supports_h1


def test_resource_trend_needs_enough_languages():
    with pytest.raises(ValueError, match=">=3"):
        resource_trend([0, 1], [0.0, -0.1])


def test_h1_requires_the_ci_to_exclude_zero():
    """A negative point-estimate slope with a CI straddling zero is not evidence.

    This is the spec's 'confidence intervals, never bare point estimates' rule, enforced.
    """
    rng = np.random.default_rng(1)
    ranks = list(range(12))
    deltas = list(-0.002 * np.array(ranks) + rng.normal(0, 0.05, size=12))
    trend = resource_trend(ranks, deltas, n_resamples=500, seed=0)
    if trend.slope < 0 and trend.slope_ci.hi >= 0:
        assert not trend.supports_h1
