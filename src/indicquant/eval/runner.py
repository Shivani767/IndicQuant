"""Evaluation driver: conditions x languages x seeds, with routing capture wired in.

Two things this module is responsible for getting right:

1. TEACHER-FORCED ROUTING CAPTURE. Routing traces must come from identical token sequences
   across conditions, or `topk_agreement` will (correctly) refuse to compare them. The runner
   therefore fixes the evaluation examples ONCE per (task, language, seed) and replays exactly
   those token sequences through every condition. See ARCHITECTURE.md §6.1.

2. FIXED SAMPLING. Sampling parameters are held identical across conditions and reported
   explicitly. Quantization interacts with sampling, so an unreported temperature difference
   between conditions would be an alternative explanation for any measured gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SamplingConfig:
    """Held constant across all conditions and reported with every result."""

    temperature: float = 0.0  # greedy by default — removes sampling noise from the
    # capability comparison entirely
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 256
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass
class EvalPlan:
    """A fully enumerated evaluation sweep. Buildable without a GPU."""

    conditions: list[str]
    languages: list[str]
    tasks: list[str]
    seeds: list[int]
    n_examples: int
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    capture_routing: bool = True

    @property
    def n_runs(self) -> int:
        return len(self.conditions) * len(self.languages) * len(self.tasks) * len(self.seeds)

    def describe(self) -> str:
        return (
            f"{self.n_runs} runs = {len(self.conditions)} conditions x "
            f"{len(self.languages)} languages x {len(self.tasks)} tasks x "
            f"{len(self.seeds)} seeds, {self.n_examples} examples each "
            f"({self.n_runs * self.n_examples:,} generations)"
        )


def phase0_plan(languages: list[str]) -> EvalPlan:
    """The GO/NO-GO sweep: conditions A, B, E only.

    Five seeds, not three. The spec is explicit that seeds are the last thing to cut — narrow
    scope is forgivable, noisy claims are not — so the pilot economizes on languages and
    tasks instead.
    """
    from indicquant.eval.tasks import phase0_tasks

    return EvalPlan(
        conditions=["A", "B", "E"],
        languages=languages,
        tasks=phase0_tasks(),
        seeds=[0, 1, 2, 3, 4],
        n_examples=500,
    )


def run_evaluation(
    plan: EvalPlan,
    checkpoints: dict[str, str | Path],
    out_dir: str | Path = "results",
) -> Path:
    """Execute an evaluation plan.

    `checkpoints` maps condition ID to a local checkpoint path (Condition A uses the HF
    baseline). Requires a CUDA device.
    """
    raise NotImplementedError(
        "Phase 0 deliverable. Requires a CUDA device.\n"
        "\n"
        "Implementation order that matters:\n"
        "  1. Materialize examples ONCE per (task, language, seed) and cache the tokenized\n"
        "     input_ids. Every condition replays the SAME ids — this is what makes routing\n"
        "     traces comparable and is not an optimization.\n"
        "  2. For MCQ, score by per-choice log-likelihood, not by parsing generated text.\n"
        "  3. Wrap each forward pass in RouterRecorder when the condition sets\n"
        "     capture_routing, calling set_sequence() before every batch.\n"
        "  4. Write raw generations, not just scores — the spec commits to releasing raw\n"
        "     outputs, and script-integrity analysis needs them."
    )
