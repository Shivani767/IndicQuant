---
license: apache-2.0
base_model: sarvamai/sarvam-30b
base_model_relation: quantized
language: [en, hi, bn, ta, te, mr, gu, kn, ml, pa, or, as]
tags: [awq, int4, quantized, indic, moe, llm-compressor]
---

# sarvam-30b — INT4 AWQ, Indic-calibrated

<!-- TEMPLATE. Fill from the run manifest; do not hand-write numbers. -->

INT4 (W4A16) AWQ quantization of [`sarvamai/sarvam-30b`](https://huggingface.co/sarvamai/sarvam-30b),
calibrated on **{CALIBRATION_CORPUS}** instead of the customary English corpus.

## Why this exists

Post-training quantization fits its parameters to activation statistics measured on a
calibration corpus, and in near-universal practice that corpus is English. For a model whose
purpose is Indic-language capability, that is a mismatch — and in an MoE it is sharper still,
because experts that specialize in a language may see almost none of that language during
calibration.

This checkpoint holds the method and bit-width fixed and changes only the calibration
distribution. **{HEADLINE_RESULT}**

## Results

Measured against FP16 at identical sampling parameters, ≥5 seeds, bootstrap 95% CIs.

| Language | Tier | FP16 | INT4 (English calib.) | INT4 (this checkpoint) | Δ recovered |
|---|---|---:|---:|---:|---:|
| … | | | | | |

<!-- Also report:
     - English performance, to show the fix does not trade it away
     - script integrity (purity, drift rate, malformed-grapheme rate)
     - a held-out language never seen in calibration, for transfer -->

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{REPO_ID}", dtype="auto", device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}", trust_remote_code=True)
```

## Quantization recipe

Fully reproducible from [IndicQuant](https://github.com/Shivani767/IndicQuant):

```bash
indicquant calib build --config {CALIBRATION_CONFIG}
indicquant quantize --condition {CONDITION_ID}
```

| | |
|---|---|
| Method | AWQ (`llm-compressor` {VERSION}) |
| Scheme | W4A16, group size 128, symmetric |
| Calibration | {CALIBRATION_CORPUS}, {N_TOKENS} tokens, {N_SEQUENCES} × {SEQ_LEN} |
| Preserved at FP16 | `lm_head`, router gates, shared experts, embeddings |
| `calibrate_all_experts` | {TRUE_OR_FALSE} |
| Config hash | `{CONFIG_HASH}` |

Router gates and shared experts are deliberately left unquantized: the gates are ~0.03% of
parameters and shared experts are active for every token, so quantizing either costs accuracy
for negligible savings.

## Limitations

- Quantization degrades quality relative to FP16 in **every** language. This checkpoint
  reduces the Indic-specific portion of that loss; it does not eliminate quantization loss.
- {ANY_LANGUAGE_WITH_A_REGRESSION}
- Calibration used {N_TOKENS} tokens. Results may shift at a substantially different budget.
- Evaluated on {BENCHMARKS}. Performance on other tasks is unmeasured.

## Citation

```bibtex
@misc{indicquant2026,
  title  = {Calibration-Language Mismatch and Expert-Routing Drift in Quantized Indic MoE Models},
  author = {Bhandari, Shivani},
  year   = {2026},
  url    = {https://github.com/Shivani767/IndicQuant}
}
```
