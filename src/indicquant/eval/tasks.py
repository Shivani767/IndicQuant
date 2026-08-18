"""Benchmark adapters.

All dataset IDs verified on Hugging Face on 2026-08-18. NOTE: the spec named *IndicBench* and
*IndQA*; neither resolves to a canonical HF dataset, so the suite below replaces them. Indic
NLP moved fast through 2026 — re-verify before Phase 0.

Each task exposes the same two methods so `runner.py` stays task-agnostic:
    load(language, n, seed) -> list[Example]
    score(predictions, references) -> dict[str, float]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Example:
    id: str
    prompt: str
    reference: str
    language: str
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str
    task_type: str  # "mcq" | "generation"

    def load(self, language: str, n: int, seed: int) -> list[Example]: ...
    def score(self, predictions: list[str], examples: list[Example]) -> dict[str, float]: ...


TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "milu": {
        "hf_id": "ai4bharat/MILU",
        "task_type": "mcq",
        "languages": ["hi", "bn", "ta", "te", "mr", "gu", "kn", "pa", "or", "ml", "en"],
        "metric": "accuracy",
        "note": "Broadest native-Indic MCQ benchmark. Not translated from English, which "
        "matters: translated benchmarks measure translation artifacts as much as "
        "capability.",
    },
    "flores": {
        "hf_id": "google/IndicGenBench_flores_in",
        "task_type": "generation",
        "metric": "chrf",
        "note": "Parallel across all languages — same content everywhere. Also the source "
        "for the fertility measurement, so the two analyses share a corpus.",
    },
    "xquad": {
        "hf_id": "google/IndicGenBench_xquad_in",
        "task_type": "generation",
        "metric": "f1",
        "note": "Extractive QA — gradeable without a judge model.",
    },
    "crosssum": {
        "hf_id": "google/IndicGenBench_crosssum_in",
        "task_type": "generation",
        "metric": "chrf",
        "note": "Long-form. Where fluency loss shows up most clearly.",
    },
    "indicqa": {
        "hf_id": "ai4bharat/IndicQA",
        "task_type": "generation",
        "metric": "f1",
    },
    "indicifeval": {
        "hf_id": "ai4bharat/IndicIFEval",
        "task_type": "generation",
        "metric": "instruction_accuracy",
        "note": "Programmatically checkable constraints — immune to metric arguments.",
    },
    "codemixed": {
        "hf_id": None,
        "task_type": "generation",
        "metric": "chrf",
        "note": "CONSTRUCTED. No canonical benchmark exists (checked 2026-08-18). Held out "
        "from the code-mixed calibration corpus. Building it is a contribution; "
        "H4 predicts this degrades worst of all.",
    },
}


def load_task(name: str) -> Task:
    """Instantiate a task adapter by registry name."""
    if name not in TASK_REGISTRY:
        raise KeyError(f"unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}")
    raise NotImplementedError(
        f"Phase 0 deliverable for task {name!r}.\n"
        "Implement per-task: load() returning Example objects, score() returning a metric "
        "dict. Re-verify the HF dataset ID, config names and split names first — they were "
        "checked on 2026-08-18 and this space moves.\n"
        "For MCQ, score by comparing per-choice log-likelihood rather than by parsing "
        "generated text: parsing conflates format-following with knowledge, and quantized "
        "models degrade at format-following first."
    )


def phase0_tasks() -> list[str]:
    """The minimum task set for the GO/NO-GO plot.

    One MCQ task and one generation task. The pilot exists to produce a single plot; adding
    tasks beyond what that plot needs is how a one-week pilot becomes a three-week one.
    """
    return ["milu", "flores"]
