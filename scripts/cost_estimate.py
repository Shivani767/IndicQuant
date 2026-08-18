#!/usr/bin/env python3
"""Price a run before renting anything.

The spec's instruction is to cost the pilot precisely, then extrapolate. This computes from
the active configs rather than from a remembered number, so it re-checks itself whenever the
condition or language set changes.

Every rate below is an ASSUMPTION and is printed as such. Replace with measured throughput
after the first GPU session — the point of this script is to make the estimate falsifiable,
not to be right on the first try.

    python scripts/cost_estimate.py --phase 0 --gpu-hourly 3.50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indicquant.config import load_languages, load_model_config  # noqa: E402

# --- Assumptions. Replace with measurements after the first GPU session. ---------------
ASSUMPTIONS = {
    "quantization_hours_per_condition": 6.0,
    # 2304 expert matrices (18 layers x 128 experts) is a lot of small GEMMs; AWQ on a
    # dense 30B would be faster. Treated as the largest single uncertainty here.
    "download_hours": 1.5,          # 128.6 GB at ~250 MB/s on a cloud node
    "eval_generations_per_hour": 9000,
    "preflight_hours": 2.0,         # one FP16 pass per calibration corpus, hooks on
    "storage_gb_month": 0.10,       # USD per GB-month
    "egress_gb": 0.09,              # USD per GB, for pushing checkpoints to HF
}


def phase0_estimate(gpu_hourly: float) -> dict:
    model = load_model_config("sarvam-30b")
    langset = load_languages()

    n_languages = len(langset.phase0())
    conditions_to_quantize = 2          # B and E; A is the FP16 baseline
    n_conditions = 3
    n_seeds = 5
    n_tasks = 2
    n_examples = 500

    generations = n_conditions * n_languages * n_tasks * n_seeds * n_examples
    eval_hours = generations / ASSUMPTIONS["eval_generations_per_hour"]
    quant_hours = conditions_to_quantize * ASSUMPTIONS["quantization_hours_per_condition"]
    gpu_hours = (
        ASSUMPTIONS["download_hours"]
        + ASSUMPTIONS["preflight_hours"]
        + quant_hours
        + eval_hours
    )

    fp = model["footprint"]
    disk_gb = fp["disk_fp32_gb"] + fp["disk_bf16_gb"] + conditions_to_quantize * fp["disk_int4_gb"]

    gpu_cost = gpu_hours * gpu_hourly
    storage_cost = disk_gb * ASSUMPTIONS["storage_gb_month"] * 0.5   # ~2 weeks
    overhead = 0.30                     # false starts, OOM retries, a failed quantization

    return {
        "n_languages": n_languages,
        "n_conditions": n_conditions,
        "n_seeds": n_seeds,
        "generations": generations,
        "download_hours": ASSUMPTIONS["download_hours"],
        "preflight_hours": ASSUMPTIONS["preflight_hours"],
        "quant_hours": quant_hours,
        "eval_hours": eval_hours,
        "gpu_hours": gpu_hours,
        "gpu_cost": gpu_cost,
        "storage_gb": disk_gb,
        "storage_cost": storage_cost,
        "subtotal": gpu_cost + storage_cost,
        "total_with_overhead": (gpu_cost + storage_cost) * (1 + overhead),
        "overhead_pct": overhead,
    }


def phase1_estimate(gpu_hourly: float) -> dict:
    langset = load_languages()
    n_languages = len(langset)
    conditions_to_quantize = 5          # B, C, D, E, F (+G shares the pipeline)
    n_conditions = 7
    n_seeds = 5
    n_tasks = 4
    n_examples = 500

    generations = n_conditions * n_languages * n_tasks * n_seeds * n_examples
    eval_hours = generations / ASSUMPTIONS["eval_generations_per_hour"]
    quant_hours = conditions_to_quantize * ASSUMPTIONS["quantization_hours_per_condition"]
    gpu_hours = quant_hours + eval_hours

    model = load_model_config("sarvam-30b")
    fp = model["footprint"]
    disk_gb = fp["disk_fp32_gb"] + fp["disk_bf16_gb"] + 5 * fp["disk_int4_gb"] + fp["disk_int8_gb"]

    gpu_cost = gpu_hours * gpu_hourly
    storage_cost = disk_gb * ASSUMPTIONS["storage_gb_month"]
    overhead = 0.30
    return {
        "n_languages": n_languages,
        "n_conditions": n_conditions,
        "generations": generations,
        "quant_hours": quant_hours,
        "eval_hours": eval_hours,
        "gpu_hours": gpu_hours,
        "gpu_cost": gpu_cost,
        "storage_gb": disk_gb,
        "storage_cost": storage_cost,
        "subtotal": gpu_cost + storage_cost,
        "total_with_overhead": (gpu_cost + storage_cost) * (1 + overhead),
        "overhead_pct": overhead,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpu-hourly", type=float, default=3.50,
                        help="USD/hour (H200 141GB ~ $3.50, H100 80GB ~ $2.50)")
    args = parser.parse_args()

    est = phase0_estimate(args.gpu_hourly) if args.phase == 0 else phase1_estimate(args.gpu_hourly)

    print()
    print(f"  IndicQuant — Phase {args.phase} cost estimate")
    print(f"  {'=' * 52}")
    print(f"  GPU rate                 ${args.gpu_hourly:.2f}/hour")
    print(f"  Languages                {est['n_languages']}")
    print(f"  Conditions               {est['n_conditions']}")
    print(f"  Generations              {est['generations']:,}")
    print()
    if "download_hours" in est:
        print(f"  download (128.6 GB)      {est['download_hours']:>6.1f} h")
        print(f"  expert-coverage preflight{est['preflight_hours']:>6.1f} h")
    print(f"  quantization             {est['quant_hours']:>6.1f} h")
    print(f"  evaluation               {est['eval_hours']:>6.1f} h")
    print(f"  {'-' * 52}")
    print(f"  GPU hours                {est['gpu_hours']:>6.1f} h   ${est['gpu_cost']:>8.2f}")
    print(f"  storage ({est['storage_gb']:.0f} GB)      {'':>8}   ${est['storage_cost']:>8.2f}")
    print(f"  {'-' * 52}")
    print(f"  subtotal                 {'':>8}   ${est['subtotal']:>8.2f}")
    print(f"  + {est['overhead_pct']:.0%} overhead        {'':>8}   "
          f"${est['total_with_overhead']:>8.2f}")
    print()
    print("  Assumptions (replace with measurements after the first GPU session):")
    for key, value in ASSUMPTIONS.items():
        print(f"    {key:<36} {value}")
    print()
    print("  Control: cut languages before cutting seeds. Narrow scope is forgivable;")
    print("  noisy claims are not.")
    print()


if __name__ == "__main__":
    main()
