#!/usr/bin/env bash
# Phase 0 pilot — the GO/NO-GO run.
#
# Conditions A (FP16), B (AWQ-INT4/English), E (AWQ-INT4/Indic).
# Five languages: Hindi, Tamil, Odia, Hinglish, English. Five seeds.
#
# SINGLE DELIVERABLE: one plot — Δ-accuracy vs. language resource level, English- vs.
# Indic-calibrated.
#
#   Gap visible  -> GO.
#   No gap       -> STOP. Publish the null result. It cost a week and it is useful.
#
# Build nothing beyond what this plot requires.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/sarvam-30b}"
RESULTS="${RESULTS:-./results/phase0}"

echo "==> 0. sanity: architecture, parameter budget, cost"
indicquant info
indicquant cost --phase 0

echo
echo "==> 1. calibration corpora (equal token budgets — the invariant)"
indicquant calib build --config english_c4
indicquant calib build --config indic_sangraha

echo
echo "==> 2. expert-coverage pre-flight"
# One FP16 forward pass per corpus with router hooks on. No quantization.
# If English calibration starves the experts Indic text relies on, H1 is predicted here,
# before any checkpoint exists. This is the cheapest de-risking available.
indicquant evaluate --conditions A --phase 0 --dry-run
echo "    [preflight: see routing.metrics.expert_coverage — Phase C]"

echo
echo "==> 3. quantize B and E (identical except calibration distribution)"
indicquant quantize --condition B
indicquant quantize --condition E

echo
echo "==> 4. evaluate A, B, E x 5 languages x 5 seeds"
indicquant evaluate --conditions A,B,E --phase 0

echo
echo "==> 5. the plot"
echo "    ${RESULTS}/degradation_vs_resource.png"
echo
echo "    Gap visible -> GO to Phase 1."
echo "    No gap      -> STOP and publish the null result."
