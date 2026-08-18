"""Routing metric tests.

The important one is `test_topk_agreement_rejects_mismatched_sequences`. The teacher-forcing
guard (ARCHITECTURE.md §6.1) is the difference between a real Phase 2 result and a retracted
one, so it is asserted here rather than trusted to discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicquant.routing.metrics import (
    RoutingComparisonError,
    expert_activation_histogram,
    expert_coverage,
    language_affinity_matrix,
    router_entropy,
    topk_agreement,
)

# -- the teacher-forcing guard ----------------------------------------------------------


def test_topk_agreement_rejects_mismatched_sequences(synthetic_trace):
    """Comparing routing across different token sequences must raise, not return a number.

    On free generation the two models' sequences diverge, and every routing difference
    afterwards is sequence divergence rather than quantization. A function that silently
    returned a plausible number here would produce a fake result.
    """
    other = synthetic_trace.copy()
    other["fingerprint"] = "DIFFERENT"
    other["condition"] = "B"

    with pytest.raises(RoutingComparisonError, match="fingerprint"):
        topk_agreement(synthetic_trace, other)


def test_topk_agreement_requires_fingerprint_column(synthetic_trace):
    no_fp = synthetic_trace.drop(columns=["fingerprint"])
    with pytest.raises(RoutingComparisonError, match="fingerprint"):
        topk_agreement(synthetic_trace, no_fp)


def test_topk_agreement_non_strict_skips_the_guard(synthetic_trace):
    """strict=False exists for exploratory work only and must never back a reported result."""
    other = synthetic_trace.copy()
    other["fingerprint"] = "DIFFERENT"
    report = topk_agreement(synthetic_trace, other, strict=False)
    assert report.n_positions > 0


# -- agreement arithmetic ---------------------------------------------------------------


def test_self_agreement_is_one(synthetic_trace):
    report = topk_agreement(synthetic_trace, synthetic_trace)
    assert np.allclose(report.overlap["overlap"], 1.0)
    assert np.allclose(report.rank_weighted["rank_weighted"], 1.0)
    assert np.allclose(report.top1_agreement["top1"], 1.0)


def test_disjoint_routing_gives_zero_agreement(synthetic_trace):
    other = synthetic_trace.copy()
    other["expert_ids"] = other["expert_ids"].apply(lambda ids: [i + 100 for i in ids])
    report = topk_agreement(synthetic_trace, other)
    assert np.allclose(report.overlap["overlap"], 0.0)
    assert np.allclose(report.top1_agreement["top1"], 0.0)


def test_partial_overlap_is_scored_proportionally(synthetic_trace):
    other = synthetic_trace.copy()
    other["expert_ids"] = other["expert_ids"].apply(lambda ids: [ids[0], 99])
    report = topk_agreement(synthetic_trace, other)
    assert np.allclose(report.overlap["overlap"], 0.5)  # 1 of 2 retained
    assert np.allclose(report.top1_agreement["top1"], 1.0)  # rank-1 unchanged


def test_rank_weighted_penalises_top1_changes_more(synthetic_trace):
    """A changed rank-1 expert must cost more than a changed rank-2 expert."""
    lost_first = synthetic_trace.copy()
    lost_first["expert_ids"] = lost_first["expert_ids"].apply(lambda ids: [99, ids[1]])
    lost_second = synthetic_trace.copy()
    lost_second["expert_ids"] = lost_second["expert_ids"].apply(lambda ids: [ids[0], 99])

    a = topk_agreement(synthetic_trace, lost_first).rank_weighted["rank_weighted"].mean()
    b = topk_agreement(synthetic_trace, lost_second).rank_weighted["rank_weighted"].mean()
    assert a < b


def test_no_overlapping_positions_raises(synthetic_trace):
    other = synthetic_trace.copy()
    other["position"] = other["position"] + 1000
    with pytest.raises(RoutingComparisonError, match="overlapping"):
        topk_agreement(synthetic_trace, other)


# -- coverage ---------------------------------------------------------------------------


def test_expert_coverage_counts_tokens(synthetic_trace):
    report = expert_coverage(synthetic_trace, num_experts=8)
    assert report.counts.shape == (2, 8)  # 2 layers x 8 experts
    # 2 sequences x 4 positions x 2 layers x 2 experts = 32 expert activations
    assert report.counts.to_numpy().sum() == 32


def test_starved_experts_finds_unused_ones(synthetic_trace):
    report = expert_coverage(synthetic_trace, num_experts=8)
    starved = report.starved_experts(threshold=1)
    # Experts 2, 3, 6, 7 are never routed to, in both layers.
    assert len(starved) == 8
    assert set(starved["expert_ids"].unique()) == {2, 3, 6, 7}


def test_gini_detects_concentration(synthetic_trace):
    """Only 4 of 8 experts are used, so routing is measurably concentrated."""
    report = expert_coverage(synthetic_trace, num_experts=8)
    gini = report.gini()
    assert (gini > 0.4).all()


def test_activation_histogram_normalizes(synthetic_trace):
    hist = expert_activation_histogram(synthetic_trace, normalize=True)
    assert np.allclose(hist.sum(axis=1), 1.0)


def test_activation_histogram_filters_by_language(synthetic_trace):
    hi = expert_activation_histogram(synthetic_trace, language="hi", normalize=False)
    assert hi[0].sum() > 0  # Hindi routes to expert 0
    assert hi.get(4, pd.Series(0)).sum() == 0  # never to expert 4


# -- entropy ----------------------------------------------------------------------------


def test_router_entropy_uniform_logits_is_maximal(synthetic_trace):
    """Uniform logits over 8 experts give log2(8) = 3 bits."""
    result = router_entropy(synthetic_trace)
    assert np.allclose(result["entropy_mean"], 3.0, atol=1e-6)


def test_router_entropy_peaked_logits_is_lower(synthetic_trace):
    peaked = synthetic_trace.copy()
    peaked["logits"] = peaked["logits"].apply(lambda _: [10.0] + [-10.0] * 7)
    assert router_entropy(peaked)["entropy_mean"].mean() < 1.0


def test_router_entropy_without_logits_raises(synthetic_trace):
    no_logits = synthetic_trace.copy()
    no_logits["logits"] = None
    with pytest.raises(ValueError, match="logits"):
        router_entropy(no_logits)


# -- specialization ---------------------------------------------------------------------


def test_affinity_detects_planted_specialization(synthetic_trace):
    """Experts 0/1 fire only on Hindi and 4/5 only on English, so this is maximal."""
    report = language_affinity_matrix(synthetic_trace, layer=1, n_permutations=50, seed=0)
    assert report.specialization == pytest.approx(1.0)
    assert report.matrix.shape[1] == 2


def test_affinity_permutation_null_rejects_unspecialized_routing():
    """Routing uncorrelated with language must not read as specialized.

    This is the control that stops the affinity matrix from asserting specialization that
    is really just unequal token counts.
    """
    rng = np.random.default_rng(0)
    rows = []
    for seq_id in range(40):
        language = "hi" if seq_id % 2 else "en"
        for position in range(10):
            rows.append(
                {
                    "condition": "A",
                    "language": language,
                    "layer": 1,
                    "seq_id": seq_id,
                    "position": position,
                    "fingerprint": f"fp{seq_id}",
                    "expert_ids": sorted(rng.choice(8, size=2, replace=False).tolist()),
                    "expert_weights": [0.6, 0.4],
                    "logits": [0.1] * 8,
                }
            )
    trace = pd.DataFrame(rows)
    report = language_affinity_matrix(trace, layer=1, n_permutations=100, seed=0)
    assert not report.is_specialized, f"z={report.z_score:.2f} on random routing"
