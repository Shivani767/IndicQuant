"""Figures.

The GO/NO-GO plot is `degradation_vs_resource()`. The spec is explicit that Phase 0 has a
single deliverable — one plot — and that nothing should be built beyond what that plot
requires. This module is written in that order.

Style is deliberately plain: matplotlib defaults, no seaborn dependency, readable in
greyscale. These figures go into a blog post and possibly a paper, where legibility beats
decoration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: these run on rented nodes and in CI
import matplotlib.pyplot as plt  # noqa: E402

TIER_MARKERS = {
    "control": ("o", "#444444"),
    "high": ("s", "#1b7837"),
    "medium": ("^", "#4575b4"),
    "low": ("v", "#d73027"),
    "code_mixed": ("D", "#762a83"),
}


def degradation_vs_resource(
    results: list[dict[str, Any]],
    out_path: str | Path,
    title: str = "Quantization degradation vs. language resource level",
) -> Path:
    """THE PHASE 0 PLOT — the GO/NO-GO deliverable.

    x: resource rank (0 = English control, rising as resources fall)
    y: Δ accuracy from FP16 (negative = degradation)
    two series: English-calibrated (Condition B) and Indic-calibrated (Condition E)

    Each `results` entry needs: language, language_name, tier, resource_rank, condition,
    delta, ci_lo, ci_hi.

    Reading it: if the English-calibrated series slopes down and the Indic-calibrated series
    is flatter, H1 holds and the fix works. If the two series are indistinguishable, that is
    the null result — publish it and stop, exactly as the spec prescribes.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for condition, label, color, style in (
        ("B", "English-calibrated (standard practice)", "#d73027", "-"),
        ("E", "Indic-calibrated", "#1b7837", "--"),
    ):
        series = sorted(
            (r for r in results if r["condition"] == condition), key=lambda r: r["resource_rank"]
        )
        if not series:
            continue
        x = [r["resource_rank"] for r in series]
        y = [r["delta"] for r in series]
        lo = [r["delta"] - r.get("ci_lo", r["delta"]) for r in series]
        hi = [r.get("ci_hi", r["delta"]) - r["delta"] for r in series]
        ax.errorbar(
            x,
            y,
            yerr=[lo, hi],
            label=label,
            color=color,
            linestyle=style,
            marker="o",
            capsize=3,
            linewidth=1.8,
            markersize=6,
        )

    for r in sorted({r["language"]: r for r in results}.values(), key=lambda r: r["resource_rank"]):
        ax.annotate(
            r.get("language_name", r["language"]),
            (r["resource_rank"], r["delta"]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7.5,
            color="#555555",
        )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Language resource level  (0 = English control → higher = lower resource)")
    ax.set_ylabel("Δ accuracy vs. FP16")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25, linestyle=":")
    # The ordinal caveat travels with the figure rather than living only in the caption.
    fig.text(
        0.01,
        0.01,
        "Resource level is an ordinal rank; Sarvam has not published per-language "
        "pretraining shares.",
        fontsize=6.5,
        color="#777777",
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fertility_by_language(results: list[Any], out_path: str | Path) -> Path:
    """Tokenizer fertility per language — the Phase A figure, produced without a GPU.

    Accepts `FertilityResult` objects or plain dicts.
    """
    rows = [r if isinstance(r, dict) else r.to_dict() for r in results]
    rows.sort(key=lambda r: r["tokens_per_word"])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["language_name"] for r in rows]
    values = [r["tokens_per_word"] for r in rows]
    colors = [TIER_MARKERS.get(r["tier"], ("o", "#888888"))[1] for r in rows]

    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Tokens per word (FLORES-200 parallel text)")
    ax.set_title("Tokenizer fertility — sarvam-30b")
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    english = next((r for r in rows if r["language"] == "en"), None)
    if english:
        ax.axvline(
            english["tokens_per_word"],
            color="#444444",
            linestyle="--",
            linewidth=1,
            label="English baseline",
        )
        ax.legend(frameon=False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def expert_activation_heatmap(
    matrix: Any, out_path: str | Path, title: str = "Expert activation by language"
) -> Path:
    """Expert x language affinity heatmap for one layer (Phase 2).

    `matrix` is the DataFrame from `routing.metrics.language_affinity_matrix`.
    """
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Expert index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(language | expert activated)")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def routing_agreement_by_language(
    agreement_by_language: dict[str, float], out_path: str | Path
) -> Path:
    """Top-k routing agreement with FP16, per language (Phase 2, H2).

    H2 predicts agreement falls furthest for low-resource languages.
    """
    items = sorted(agreement_by_language.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4575b4")
    ax.set_xlabel("Top-k expert agreement with FP16")
    ax.set_xlim(0, 1)
    ax.set_title("Routing agreement under INT4 quantization")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def expert_coverage_comparison(
    coverage_by_corpus: dict[str, Any], out_path: str | Path, layer: int
) -> Path:
    """Tokens-per-expert under English vs. Indic calibration — the Phase C pre-flight figure.

    This is the cheapest plot in the project and potentially the most informative: it is
    produced from one FP16 forward pass per corpus, with no quantization at all. If the
    English curve leaves a tail of experts near zero that the Indic curve populates, H1 has
    a mechanism before any checkpoint exists.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, report in coverage_by_corpus.items():
        counts = report.counts.loc[layer].sort_values(ascending=False).to_numpy()
        ax.plot(range(len(counts)), counts, label=name, linewidth=1.6)

    ax.set_yscale("symlog")
    ax.set_xlabel("Expert (sorted by token count, descending)")
    ax.set_ylabel("Calibration tokens routed to expert")
    ax.set_title(f"Expert coverage by calibration corpus — layer {layer}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
