# Research log

Append-only. Dated entries, newest last. Negative results and dead ends get recorded here
with the same care as positive ones — the point of the log is that a decision made in week 9
can be traced back to the evidence that motivated it in week 1.

---

## 2026-08-18 — Verification pass and repo scaffold

Verified the spec's assumptions against Hugging Face and GitHub before building anything.
Several were wrong.

**Toolchain**
- **AutoAWQ is archived** (`archived: true`, last push 2025-05-11). The spec's plan to
  contribute Indic calibration support upstream to AutoAWQ is dead.
- `llm-compressor` 0.13.0 (released 2026-08-11) and `GPTQModel` v7.3.2 are both actively
  maintained. Primary path is now llm-compressor; GPTQModel is the independent cross-check.
- `llm-compressor` ships `src/llmcompressor/modeling/moe/` with per-architecture adapters
  (llama4, granitemoe, cohere2_moe, deepseekv32, …) and a documented extension tutorial at
  `docs/developer-tutorials/add-moe-support.md`. **There is no `sarvam_moe` adapter.**
  → sarvam-30b cannot currently be AWQ-quantized with the standard toolchain. Writing that
  adapter is both a prerequisite for this project and its clearest upstream contribution.

**Model**
- `sarvamai/sarvam-30b` confirmed: Apache-2.0, `SarvamMoEForCausalLM`, 32,152,650,368 params,
  **128.6 GB on disk** (fp32, 26 shards, 7122 tensors).
- 19 layers, `first_k_dense_replace=1` → layer 0 dense, layers 1–18 MoE.
- **128 routed experts, top-6, 1 shared expert**, `moe_intermediate_size=1024`.
- Router: sigmoid scoring, learned `expert_bias`, `routed_scaling_factor=2.5`,
  `router_dtype=fp32`.
- Attention names are **fused** (`attention.query_key_value`, `attention.dense`), not
  `q_proj/k_proj/v_proj/o_proj`. Experts are stored **unfused** as per-expert 2D linears.
- Computed parameter budget: **routed experts are 90.2% of parameters**. Quantizing this
  model is, to first order, quantizing 2304 small expert matrices.
- `sarvam-30b-fp8` and `sarvam-30b-gguf` already exist. INT4 + calibration-language choice
  is still open, but the clock is real.

**The key code fact.** `SarvamMoEGate.forward` returns `(topk_idx, topk_weight, logits)`.
One `register_forward_hook` per MoE layer captures the entire RQ4 surface. No model surgery
needed. Confirmed working end-to-end on CPU against the real remote code.

**Benchmarks.** The spec's *IndicBench* and *IndQA* do not resolve as canonical HF datasets.
Replaced with `ai4bharat/MILU`, `google/IndicGenBench_{flores,xquad,xorqa,crosssum}_in`,
`ai4bharat/{IndicQA,IndicIFEval,IN22-Gen}`. No canonical romanized code-mixed generation
benchmark exists — building one remains a contribution, as the spec predicted.

**InferLite.** `backend/quantization/engine.py` is a stub returning a hardcoded
`compression_ratio: 4.0`; the README states all results are "simulation mode". It cannot host
real experiments. IndicQuant is standalone.

### New conditions added (H and I)

Reading llm-compressor's MoE tutorial produced a sharper mechanism than the spec's. The
framework acknowledges that small calibration sets leave some experts barely activated, and
works around it by routing all tokens through all experts (`calibrate_all_experts`).

That fixes the token-**count** problem and not the **language** problem: under English
calibration, a Tamil-specialized expert now receives plenty of activations, all of them
English. Its scales are fit to a distribution it will never see.

- **Condition H** — routing-faithful vs. `calibrate_all_experts`, crossed with
  English/Indic calibration. Separates coverage starvation from distribution mismatch.
- **Condition I** — router gate quantized vs. preserved in FP16. Separates direct router
  perturbation from error propagating into router inputs from the layer below. The gate is
  0.03% of parameters, so preserving it is nearly free — if it recovers routing agreement,
  that is an immediately deployable recommendation.

Also added: an **expert-coverage pre-flight** that predicts H1 from one FP16 forward pass per
calibration corpus, with no quantization at all.

### Environment findings (from actually building it)

- **`llm-compressor` and `vllm` cannot coexist.** llm-compressor 0.13.0 needs
  `compressed-tensors>=0.18.0` / `transformers>=5.9.0`; every vllm release pins
  `compressed-tensors<=0.17.0` / `transformers<5`. uv's resolver proves them incompatible.
  → Two venvs on the GPU node, declared as conflicting extras. Finding this on a rented GPU
  would have cost a day.
- **`transformers` v5 works with sarvam's remote code.** Flagged in the plan as the likeliest
  early blocker; it is not one. Built the tiny model and ran a forward pass under
  transformers 5.15.0 — deprecation warnings, no errors. Risk closed.
- **Bug caught by the test suite**: orphan-combining-mark detection used
  `unicodedata.combining()`, which returns 0 for *spacing* combining marks (category `Mc`) —
  which is what most Indic vowel signs are. It would have silently under-reported the most
  common malformation in Devanagari, Tamil and Bengali. Now keyed on category.

### Still open

- **Literature check not done.** The arXiv API returned empty from the sandbox used here, so
  this must be run from an unrestricted network before any GPU spend. This is the spec's
  Week-1 item and it remains the one blocking task.
- Dense multilingual control model not yet pinned (`configs/model/control-dense.yaml`).
- Whether `convert_hf_to_gguf.py` supports `sarvam_moe` (Condition D). `sarvam-30b-gguf`
  exists, so a path does, but it may live in a fork.

### Next

1. Literature check → `docs/LITERATURE.md`
2. Tokenizer fertility on FLORES-200 (no GPU) → first real result
3. Build and freeze the four calibration corpora at equal token budgets
4. Price and book the Phase C pre-flight
