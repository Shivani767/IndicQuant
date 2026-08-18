"""Routing metrics — RQ4.

These functions read recorded Parquet traces and never touch a model. Analysis is therefore
re-runnable and reviewable without a GPU, which matters when the GPU is rented by the hour.

The load-bearing rule (ARCHITECTURE.md §6.1): routing may only be compared across conditions
on IDENTICAL, teacher-forced token sequences. On free generation, the moment two models emit
different tokens the sequences diverge, and every subsequent routing difference measures
divergent context rather than quantization. `topk_agreement` enforces this by comparing
recorded sequence fingerprints and raising rather than returning a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RoutingComparisonError(RuntimeError):
    """Raised when two routing traces are not legitimately comparable.

    Almost always means the traces were captured on different token sequences — the failure
    mode that would silently turn sequence divergence into a fake routing-drift result.
    """


# --------------------------------------------------------------------------------------
# Coverage — the Phase C pre-flight
# --------------------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Tokens routed to each expert, per layer, for one calibration corpus."""

    counts: pd.DataFrame  # index: layer, columns: expert_id, values: token count
    n_tokens: int
    n_experts: int

    def starved_experts(self, threshold: int = 32) -> pd.DataFrame:
        """Experts receiving fewer than `threshold` tokens — too few to fit scales against.

        AWQ needs enough activation samples per channel to estimate magnitude; GPTQ needs
        enough to form a non-degenerate Hessian. Below a few dozen tokens both degrade to
        noise. The default is deliberately conservative: the interesting signal is the
        *relative* starvation between English and Indic corpora, not the absolute cutoff.
        """
        mask = self.counts < threshold
        return (
            mask.stack()
            .rename("starved")
            .reset_index()
            .query("starved")
            .drop(columns="starved")
            .rename(columns={"level_1": "expert_id"})
        )

    def gini(self) -> pd.Series:
        """Per-layer Gini coefficient of the token-per-expert distribution.

        0 = perfectly uniform routing, 1 = all tokens to one expert. A corpus that routes
        unevenly leaves most experts under-calibrated regardless of its language.
        """
        out = {}
        for layer, row in self.counts.iterrows():
            x = np.sort(row.to_numpy(dtype=float))
            n = len(x)
            total = x.sum()
            if total <= 0:
                out[layer] = float("nan")
                continue
            index = np.arange(1, n + 1)
            out[layer] = float((2 * index - n - 1).dot(x) / (n * total))
        return pd.Series(out, name="gini")


def expert_coverage(trace: pd.DataFrame, num_experts: int | None = None) -> CoverageReport:
    """Count tokens routed to each expert, per layer.

    THE PHASE C PRE-FLIGHT (ARCHITECTURE.md §7). Run one FP16 forward pass over each
    calibration corpus with router hooks on, then call this. If English calibration starves
    experts that Indic text relies on, H1 is predicted before a single checkpoint is
    quantized — a few GPU-hours instead of a few GPU-days to de-risk the whole project.
    """
    exploded = trace[["layer", "expert_ids"]].explode("expert_ids")
    exploded["expert_ids"] = exploded["expert_ids"].astype(int)
    counts = exploded.groupby(["layer", "expert_ids"]).size().unstack(fill_value=0).sort_index()
    n_experts = num_experts or int(counts.columns.max()) + 1
    counts = counts.reindex(columns=range(n_experts), fill_value=0)
    return CoverageReport(counts=counts, n_tokens=len(trace), n_experts=n_experts)


# --------------------------------------------------------------------------------------
# Activation distribution
# --------------------------------------------------------------------------------------


def expert_activation_histogram(
    trace: pd.DataFrame, language: str | None = None, normalize: bool = True
) -> pd.DataFrame:
    """Per-layer distribution over experts, optionally restricted to one language.

    Returns a (layer x expert) frame. With `normalize=True` each row sums to 1, making
    distributions comparable across languages with different token counts — which they
    always have, because tokenizer fertility differs sharply across Indic scripts.
    """
    if language is not None:
        trace = trace[trace["language"] == language]
    report = expert_coverage(trace)
    counts = report.counts
    if normalize:
        totals = counts.sum(axis=1).replace(0, np.nan)
        counts = counts.div(totals, axis=0).fillna(0.0)
    return counts


# --------------------------------------------------------------------------------------
# Agreement — the headline RQ4 metric
# --------------------------------------------------------------------------------------


@dataclass
class AgreementReport:
    overlap: pd.DataFrame  # per-layer mean top-k overlap fraction
    rank_weighted: pd.DataFrame  # per-layer mean rank-weighted agreement
    top1_agreement: pd.DataFrame  # per-layer fraction where the rank-1 expert matches
    n_positions: int

    def summary(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.overlap.rename(columns={self.overlap.columns[0]: "overlap"}),
                self.rank_weighted.rename(columns={self.rank_weighted.columns[0]: "rank_weighted"}),
                self.top1_agreement.rename(columns={self.top1_agreement.columns[0]: "top1"}),
            ],
            axis=1,
        )


def topk_agreement(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    strict: bool = True,
) -> AgreementReport:
    """Compare routing between two conditions on identical token sequences.

    `reference` is normally the FP16 (Condition A) trace; `comparison` a quantized one.

    THE TEACHER-FORCING GUARD (ARCHITECTURE.md §6.1). Both traces must come from the same
    forced token sequences. This function verifies that via the recorded per-sequence
    fingerprints and raises `RoutingComparisonError` on any mismatch. Comparing routing
    across freely-generated sequences produces a number that looks like routing drift and is
    actually sequence divergence — the single most likely way to make Phase 2 unpublishable.

    `strict=False` is provided only for exploratory work on partial traces and must never be
    used for a reported result.

    Metrics, all per layer:
      - overlap:       |A ∩ B| / k, unordered set agreement
      - rank_weighted: agreement weighted by 1/log2(rank+1), so a changed top-1 costs more
                       than a changed 6th choice — which matches the effect on the output,
                       since routing weights are steeply ordered
      - top1:          fraction of positions where the highest-weighted expert is unchanged
    """
    _assert_comparable(reference, comparison, strict=strict)

    key = ["layer", "seq_id", "position"]
    merged = reference.merge(
        comparison, on=key, suffixes=("_ref", "_cmp"), how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise RoutingComparisonError(
            "no overlapping (layer, seq_id, position) rows between the two traces — they "
            "were captured over different data, not different conditions."
        )

    ref_experts = merged["expert_ids_ref"].to_numpy()
    cmp_experts = merged["expert_ids_cmp"].to_numpy()

    overlap = np.empty(len(merged), dtype=float)
    rank_w = np.empty(len(merged), dtype=float)
    top1 = np.empty(len(merged), dtype=float)

    for i, (a, b) in enumerate(zip(ref_experts, cmp_experts, strict=True)):
        a_list = list(a)
        b_set = set(b)
        k = len(a_list)
        hits = sum(1 for e in a_list if e in b_set)
        overlap[i] = hits / k if k else np.nan

        # Rank-weighted: credit each reference expert by its own rank discount if retained.
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        gained = np.array([1.0 if e in b_set else 0.0 for e in a_list])
        rank_w[i] = float((gained * discounts).sum() / discounts.sum()) if k else np.nan

        top1[i] = float(bool(a_list) and bool(len(b)) and a_list[0] == b[0])

    merged = merged.assign(_overlap=overlap, _rank_w=rank_w, _top1=top1)
    grouped = merged.groupby("layer")
    return AgreementReport(
        overlap=grouped[["_overlap"]].mean().rename(columns={"_overlap": "overlap"}),
        rank_weighted=grouped[["_rank_w"]].mean().rename(columns={"_rank_w": "rank_weighted"}),
        top1_agreement=grouped[["_top1"]].mean().rename(columns={"_top1": "top1"}),
        n_positions=len(merged),
    )


def _assert_comparable(reference: pd.DataFrame, comparison: pd.DataFrame, strict: bool) -> None:
    if not strict:
        return
    for name, df in (("reference", reference), ("comparison", comparison)):
        if "fingerprint" not in df.columns:
            raise RoutingComparisonError(
                f"{name} trace has no `fingerprint` column; it was not captured by "
                "RouterRecorder and its token sequences cannot be verified."
            )

    ref_fps = reference.groupby("seq_id")["fingerprint"].first()
    cmp_fps = comparison.groupby("seq_id")["fingerprint"].first()
    shared = ref_fps.index.intersection(cmp_fps.index)
    if len(shared) == 0:
        raise RoutingComparisonError("traces share no sequence IDs")

    mismatched = shared[ref_fps.loc[shared].to_numpy() != cmp_fps.loc[shared].to_numpy()]
    if len(mismatched) > 0:
        raise RoutingComparisonError(
            f"{len(mismatched)} of {len(shared)} sequences have different token fingerprints "
            f"(e.g. seq_id={list(mismatched[:5])}). Routing was captured on different token "
            "sequences, so any 'drift' measured here would be sequence divergence, not "
            "quantization. Re-run both conditions teacher-forced on identical inputs — see "
            "ARCHITECTURE.md §6.1."
        )


# --------------------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------------------


def router_entropy(trace: pd.DataFrame, base: float = 2.0) -> pd.DataFrame:
    """Per-layer mean entropy of the router distribution.

    Sarvam's gate uses SIGMOID scoring, not softmax (`score_function: sigmoid` in
    config.json), so scores do not sum to 1. We normalize them to a distribution before
    taking entropy. This is a monotone summary of routing confidence, not a probability
    claim, and is reported as such.

    H2 predicts entropy rises under quantization — the router becomes less certain —
    disproportionately for low-resource languages.
    """
    rows = trace[trace["logits"].notna()]
    if rows.empty:
        raise ValueError(
            "no rows with stored logits. Entropy needs full router logits; either "
            "store_logits was False or logit_stride skipped every position."
        )

    out: dict[int, list[float]] = {}
    for layer, group in rows.groupby("layer"):
        vals = []
        for logits in group["logits"]:
            arr = np.asarray(logits, dtype=np.float64)
            scores = 1.0 / (1.0 + np.exp(-arr))  # sigmoid, matching the gate
            total = scores.sum()
            if total <= 0:
                continue
            p = scores / total
            p = p[p > 0]
            vals.append(float(-(p * (np.log(p) / np.log(base))).sum()))
        out[int(layer)] = vals

    return pd.DataFrame(
        {
            "layer": list(out),
            "entropy_mean": [float(np.mean(v)) if v else np.nan for v in out.values()],
            "entropy_std": [float(np.std(v)) if v else np.nan for v in out.values()],
            "n": [len(v) for v in out.values()],
        }
    ).set_index("layer")


# --------------------------------------------------------------------------------------
# Language specialization
# --------------------------------------------------------------------------------------


@dataclass
class AffinityReport:
    matrix: pd.DataFrame  # (expert_id x language), rows sum to 1
    layer: int
    specialization: float  # observed mean max-affinity
    null_mean: float  # permutation-null mean
    null_std: float
    z_score: float

    @property
    def is_specialized(self) -> bool:
        """Whether specialization exceeds the permutation null by a conventional margin."""
        return bool(np.isfinite(self.z_score) and self.z_score > 3.0)


def language_affinity_matrix(
    trace: pd.DataFrame,
    layer: int,
    n_permutations: int = 200,
    seed: int = 0,
) -> AffinityReport:
    """Per-expert language affinity for one layer, tested against a permutation null.

    P(language | expert activated), so each expert's row sums to 1. An expert that fires
    equally on all languages has a flat row; a specialized expert has a peaked one.

    THE NULL MATTERS. Some apparent specialization arises purely from unequal token counts
    per language and from routing imbalance. The permutation null shuffles language labels
    across tokens, holding both the routing structure and the label marginals fixed, and
    re-measures. Reporting a raw affinity matrix without this null would assert
    specialization rather than test it — and a reviewer would ask for exactly this.

    Note that this measures *script* affinity as much as language affinity, since Indic
    scripts are disjoint. The discriminating evidence is whether affinity persists on
    romanized code-mixed text, which shares the Latin script with English
    (ARCHITECTURE.md §6.2).
    """
    layer_trace = trace[trace["layer"] == layer]
    if layer_trace.empty:
        raise ValueError(f"no rows for layer {layer}")

    exploded = layer_trace[["language", "expert_ids"]].explode("expert_ids")
    exploded["expert_ids"] = exploded["expert_ids"].astype(int)

    observed = _affinity(exploded)
    observed_stat = float(observed.max(axis=1).mean())

    rng = np.random.default_rng(seed)
    languages = exploded["language"].to_numpy()
    null_stats = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = exploded.assign(language=rng.permutation(languages))
        null_stats[i] = float(_affinity(shuffled).max(axis=1).mean())

    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std())
    z = (observed_stat - null_mean) / null_std if null_std > 0 else float("inf")

    return AffinityReport(
        matrix=observed,
        layer=layer,
        specialization=observed_stat,
        null_mean=null_mean,
        null_std=null_std,
        z_score=float(z),
    )


def _affinity(exploded: pd.DataFrame) -> pd.DataFrame:
    counts = exploded.groupby(["expert_ids", "language"]).size().unstack(fill_value=0).sort_index()
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)
