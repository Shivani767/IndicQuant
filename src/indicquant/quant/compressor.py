"""llm-compressor driver — the primary quantization path.

AWQ (Conditions B, E, F, G, H, I) and GPTQ (Condition C) both run through here.

Everything a condition varies — method, bit-width, calibration corpus, ignore list,
`calibrate_all_experts` — is resolved from config, never hardcoded. That is what keeps
"B and E differ only in calibration distribution" a property of the code rather than a
promise in a README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indicquant.config import Manifest, config_hash, load_model_config
from indicquant.quant.sarvam_moe_adapter import AWQ_MAPPINGS, IgnorePolicy, preflight_checks


class QuantizationError(RuntimeError):
    pass


@dataclass
class QuantizationPlan:
    """Fully resolved quantization spec. Buildable and inspectable without a GPU.

    `indicquant quantize --dry-run` prints this. Being able to check what a 30-GPU-hour run
    will do, before starting it, is worth the indirection.
    """

    condition_id: str
    method: str
    bits: int
    scheme: str
    group_size: int
    symmetric: bool
    calibration: str | None
    ignore: list[str]
    targets: list[str]
    calibrate_all_experts: bool
    model_name: str
    hf_id: str
    output_dir: Path
    extra: dict[str, Any]

    def describe(self) -> str:
        lines = [
            f"Condition {self.condition_id}: {self.method.upper()} {self.scheme}",
            f"  model         : {self.hf_id}",
            f"  calibration   : {self.calibration or '(none)'}",
            f"  group_size    : {self.group_size}  symmetric: {self.symmetric}",
            f"  calibrate_all_experts: {self.calibrate_all_experts}",
            f"  targets       : {', '.join(self.targets)}",
            "  ignore        :",
            *[f"      {p}" for p in self.ignore],
            f"  output        : {self.output_dir}",
        ]
        return "\n".join(lines)


def build_plan(
    condition: dict[str, Any],
    output_root: str | Path = "checkpoints",
    grid_cell: dict[str, Any] | None = None,
) -> QuantizationPlan:
    """Resolve a condition config into an executable plan. No GPU, no model download.

    `grid_cell` selects one cell of a 2x2 for Conditions H and I, e.g.
    `{"calibration": "english_c4", "calibrate_all_experts": False}`.
    """
    cond = dict(condition)
    if grid_cell:
        cond.update(grid_cell)

    method = cond.get("method")
    if method in (None, "none"):
        raise QuantizationError(
            f"condition {cond.get('id')} is not a quantization condition (method={method!r}); "
            "Condition A is the FP16 baseline and needs no quantization step"
        )

    model_name = cond.get("model", "sarvam-30b")
    model_cfg = load_model_config(model_name)
    quant = dict(cond.get("quantization", {}))

    # Condition I swaps the whole ignore list per grid cell.
    if "ignore_variants" in cond:
        variant = (
            "gate_quantized" if grid_cell and grid_cell.get("quantize_gate") else "gate_preserved"
        )
        ignore = list(cond["ignore_variants"][variant])
    elif "ignore" in quant:
        ignore = list(quant["ignore"])
    else:
        ignore = IgnorePolicy().to_ignore_list()

    calibrate_all = quant.get("calibrate_all_experts", True)
    if grid_cell and "calibrate_all_experts" in grid_cell:
        calibrate_all = grid_cell["calibrate_all_experts"]

    bits = cond.get("bits", 4)
    scheme = quant.get("scheme", f"W{bits}A16")
    calibration = cond.get("calibration")

    cell_suffix = ""
    if grid_cell:
        cell_suffix = "_" + "_".join(f"{k}={v}" for k, v in sorted(grid_cell.items()))

    plan_key = {
        "condition": cond.get("id"),
        "method": method,
        "scheme": scheme,
        "calibration": calibration,
        "ignore": sorted(ignore),
        "calibrate_all_experts": calibrate_all,
        "group_size": quant.get("group_size", 128),
    }
    out_dir = (
        Path(output_root) / f"{model_name}-{cond.get('id')}{cell_suffix}-{config_hash(plan_key)}"
    )

    extra = {
        k: v
        for k, v in quant.items()
        if k
        not in {"scheme", "group_size", "symmetric", "targets", "ignore", "calibrate_all_experts"}
    }

    return QuantizationPlan(
        condition_id=str(cond.get("id")),
        method=method,
        bits=bits,
        scheme=scheme,
        group_size=quant.get("group_size", 128),
        symmetric=quant.get("symmetric", True),
        calibration=calibration,
        ignore=ignore,
        targets=quant.get("targets", ["Linear"]),
        calibrate_all_experts=calibrate_all,
        model_name=model_name,
        hf_id=model_cfg["hf_id"],
        output_dir=out_dir,
        extra=extra,
    )


def expand_grid(condition: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a condition's `grid` into cells (Conditions H and I).

    Cells listed in `reuse_checkpoints` are marked so the runner reuses an existing
    checkpoint rather than re-quantizing. Re-quantizing a 32B model is the dominant cost of
    the project, and the H/I grids each contain two cells that are literally Conditions B
    and E.
    """
    grid = condition.get("grid")
    if not grid:
        return [{}]

    keys = list(grid)
    cells: list[dict[str, Any]] = [{}]
    for key in keys:
        cells = [{**cell, key: value} for cell in cells for value in grid[key]]

    # YAML mapping keys must be scalars, so reuse rules are a list of {match, condition}.
    reuse = condition.get("reuse_checkpoints") or []
    normalized = {_freeze(rule["match"]): rule["condition"] for rule in reuse}
    for cell in cells:
        match = normalized.get(_freeze(cell))
        if match:
            cell["_reuse_condition"] = match
    return cells


def _freeze(obj: Any) -> tuple:
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in obj.items()))
    return obj


def run_quantization(
    plan: QuantizationPlan,
    calibration_dataset: Any,
    verify_modules: bool = True,
) -> Path:
    """Execute a plan. Requires the `quant` extra and a CUDA device.

    Imports are function-local: `llmcompressor` will not resolve on macOS/arm64, and the
    laptop dev loop must stay importable.
    """
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as exc:  # pragma: no cover - GPU-only path
        raise QuantizationError(
            "llm-compressor is not installed. On the GPU node:\n"
            "    uv venv .venv-quant && uv pip install -e '.[quant]'\n"
            "It is excluded from core deps because it will not resolve on macOS/arm64, and "
            "kept apart from '.[serve]' because llm-compressor and vllm pin incompatible "
            "compressed-tensors/transformers versions."
        ) from exc

    from indicquant.models import load_model

    model, tokenizer = load_model(plan.model_name, device_map="auto")

    if verify_modules:
        checks = preflight_checks(model, load_model_config(plan.model_name))
        if not checks["ok"]:
            raise QuantizationError(
                f"module-layout mismatch: {checks['mismatches']}. sarvam-30b is "
                "trust_remote_code, so its layout can change between Hub revisions. "
                "Re-verify quant/sarvam_moe_adapter.py patterns before quantizing."
            )

    if plan.method == "awq":
        modifier = AWQModifier(
            scheme=plan.scheme,
            targets=plan.targets,
            ignore=plan.ignore,
            mappings=AWQ_MAPPINGS,
        )
    elif plan.method == "gptq":
        modifier = GPTQModifier(
            scheme=plan.scheme,
            targets=plan.targets,
            ignore=plan.ignore,
            **{
                k: v
                for k, v in plan.extra.items()
                if k in {"actorder", "block_size", "dampening_frac"}
            },
        )
    else:
        raise QuantizationError(f"unsupported method for this backend: {plan.method}")

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=modifier,
        output_dir=str(plan.output_dir),
        max_seq_length=2048,
        num_calibration_samples=len(calibration_dataset),
    )
    tokenizer.save_pretrained(plan.output_dir)

    Manifest(
        stage="quantize",
        config={
            "condition": plan.condition_id,
            "method": plan.method,
            "scheme": plan.scheme,
            "calibration": plan.calibration,
            "ignore": plan.ignore,
            "calibrate_all_experts": plan.calibrate_all_experts,
            "group_size": plan.group_size,
        },
        outputs=[str(plan.output_dir)],
    ).write(plan.output_dir / "indicquant_manifest.json")

    return plan.output_dir
