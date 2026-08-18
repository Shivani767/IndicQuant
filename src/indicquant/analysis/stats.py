"""Statistics.

The spec's rigor requirements, implemented: >=5 seeds per configuration, confidence intervals
rather than bare point estimates, and paired comparisons where the design is paired.

`paired_delta` is the workhorse. B and E are evaluated on IDENTICAL examples, so the
comparison is paired and a paired test is both more powerful and more honest than an unpaired
one — treating paired measurements as independent would overstate the uncertainty and could
turn a real effect into a null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    level: float
    n: int
    n_resamples: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}] ({self.level:.0%} CI, n={self.n})"

    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)


def bootstrap_ci(
    values: Any,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI over per-seed (or per-example) measurements."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("no finite values to bootstrap")
    if arr.size == 1:
        v = float(arr[0])
        return BootstrapCI(mean=v, lo=v, hi=v, level=level, n=1, n_resamples=0)

    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return BootstrapCI(
        mean=float(arr.mean()),
        lo=float(np.quantile(resamples, alpha)),
        hi=float(np.quantile(resamples, 1 - alpha)),
        level=level,
        n=int(arr.size),
        n_resamples=n_resamples,
    )


@dataclass
class PairedResult:
    delta: BootstrapCI
    n_pairs: int
    p_value: float
    """Two-sided permutation p-value on the paired differences. Sign-flip permutation is the
    exact test for a paired design and makes no normality assumption — appropriate here,
    because per-example accuracy differences are bounded and discrete, not Gaussian."""

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.delta.excludes_zero()


def paired_delta(
    baseline: Any,
    treatment: Any,
    level: float = 0.95,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Paired comparison — the B vs. E test.

    Both arrays must be aligned: element i of each is the same example under two conditions.
    """
    a = np.asarray(list(baseline), dtype=float)
    b = np.asarray(list(treatment), dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs aligned arrays, got {a.shape} and {b.shape}. "
            "Conditions must be evaluated on identical examples."
        )
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        raise ValueError("no finite pairs")

    diffs = b - a
    ci = bootstrap_ci(diffs, level=level, n_resamples=n_resamples, seed=seed)

    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diffs.size))
    null = np.abs((signs * diffs).mean(axis=1))
    p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return PairedResult(delta=ci, n_pairs=int(diffs.size), p_value=p)


@dataclass
class TrendResult:
    slope: float
    intercept: float
    slope_ci: BootstrapCI
    r_squared: float
    n_languages: int

    @property
    def supports_h1(self) -> bool:
        """H1: degradation grows as resource level falls.

        With degradation encoded as a negative Δ and resource_rank increasing as resources
        fall, H1 predicts a NEGATIVE slope whose CI excludes zero.
        """
        return self.slope < 0 and self.slope_ci.hi < 0


def resource_trend(
    resource_ranks: Any,
    deltas: Any,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TrendResult:
    """Fit degradation against resource level. THE HEADLINE ANALYSIS (H1).

    `resource_ranks` is ordinal, not a measured pretraining share — Sarvam has not published
    per-language token counts. The fit is therefore reported as a rank trend, and the plot
    is labelled as such. If Sarvam ever publishes shares, swap the ordinal for the real
    values and the analysis is unchanged.

    The slope CI comes from bootstrapping over languages, so it reflects the uncertainty
    that actually matters: whether the trend would survive a different language sample.
    """
    x = np.asarray(list(resource_ranks), dtype=float)
    y = np.asarray(list(deltas), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        raise ValueError(f"need >=3 languages to fit a trend, got {x.size}")

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    slopes = np.empty(n_resamples)
    idx = np.arange(x.size)
    for i in range(n_resamples):
        pick = rng.choice(idx, size=x.size, replace=True)
        if np.ptp(x[pick]) == 0:  # degenerate resample: all one language
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(x[pick], y[pick], 1)[0]
    slopes = slopes[np.isfinite(slopes)]

    alpha = 0.025
    slope_ci = BootstrapCI(
        mean=float(slope),
        lo=float(np.quantile(slopes, alpha)),
        hi=float(np.quantile(slopes, 1 - alpha)),
        level=0.95,
        n=int(x.size),
        n_resamples=int(slopes.size),
    )
    return TrendResult(
        slope=float(slope),
        intercept=float(intercept),
        slope_ci=slope_ci,
        r_squared=float(r2),
        n_languages=int(x.size),
    )
