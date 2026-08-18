# IndicQuant — Architecture, Logic, and Implementation Guide

**Calibration-language mismatch and expert-routing drift in quantized Indic MoE models.**

This document is the working design for the project specified in
[`indic-quant-project-spec.md`](indic-quant-project-spec.md). The spec states the research
proposal; this document states what we build, why the mechanism is what it is, and how each
piece is implemented.

The spec carries a standing instruction — *verify everything on Hugging Face before designing
experiments around it.* That verification is done, and §1 records it. Several spec assumptions
turned out to be wrong in ways that change the build.

---

## Table of contents

1. [Verified ground truth](#1-verified-ground-truth)
2. [The logic: why this should happen](#2-the-logic-why-this-should-happen)
3. [Experimental design](#3-experimental-design)
4. [System architecture](#4-system-architecture)
5. [Module-by-module implementation](#5-module-by-module-implementation)
6. [Methodological guards](#6-methodological-guards)
7. [Execution phases](#7-execution-phases)
8. [Compute and cost](#8-compute-and-cost)
9. [Risks](#9-risks)

---

## 1. Verified ground truth

Everything below was checked against Hugging Face and GitHub on **2026-08-18**. Where it
contradicts the spec, this document wins.

### 1.1 Corrections to the spec

| Spec assumption | Verified reality | Consequence for the build |
|---|---|---|
| InferLite "already does runtime + quant sweeps" | [`llm-inferlite`](https://github.com/Shivani767/llm-inferlite)'s `backend/quantization/engine.py` is a stub returning a hardcoded `compression_ratio: 4.0`; its README states all results are "simulation mode" | IndicQuant is **standalone**. No real measurement can be inherited. |
| Quantize with **AutoAWQ** | AutoAWQ was **archived in May 2025** | Use **`llm-compressor`** (vLLM project, v0.13.0, released 2026-08-11) for AWQ/GPTQ/SmoothQuant. **`GPTQModel`** (v7.3.2) as the independent second implementation. |
| Benchmarks: *IndicBench*, *IndQA* | Neither resolves to a canonical HF dataset | Use `ai4bharat/MILU`, `google/IndicGenBench_{flores,xquad,xorqa,crosssum}_in`, `ai4bharat/{IndicQA,IndicIFEval,IN22-Gen}` |
| Upstream PR target: AutoAWQ | `llm-compressor` ships `src/llmcompressor/modeling/moe/` with per-architecture adapters and a `docs/developer-tutorials/add-moe-support.md` — **and no `sarvam_moe` adapter** | The upstream contribution is now obvious and necessary: **nobody can AWQ-quantize sarvam-30b today.** |
| 65K context | **131072** | Minor, but KV-cache pressure claims must use the real number |
| "Models are on HF, no gatekeeping" | True — Apache-2.0 — but the repo is **128.6 GB** (weights stored fp32 across 26 shards) | Download and disk are first-class constraints |
| "Sarvam may release quantized versions first" | Already partly true: `sarvam-30b-fp8` and `sarvam-30b-gguf` exist | INT4 + *calibration-language choice* remains open. The **finding** survives regardless of who ships checkpoints. |

### 1.2 `sarvamai/sarvam-30b` architecture

From `config.json`, `configuration_sarvam_moe.py`, and `modeling_sarvam_moe.py`:

| Property | Value |
|---|---|
| Architecture | `SarvamMoEForCausalLM` (`trust_remote_code=True`) |
| License | Apache-2.0 |
| Total parameters | 32,152,650,368 (~32.15B) |
| On-disk size | 128.6 GB (fp32, 26 safetensors shards, 7122 tensors) |
| Layers | 19, with `first_k_dense_replace=1` → **layer 0 dense, layers 1–18 MoE (18 MoE layers)** |
| Hidden size | 4096 |
| Routed experts | **128 per MoE layer**, `moe_intermediate_size=1024` |
| Experts per token | **6** (top-6) |
| Shared experts | 1 (always active) |
| Router | sigmoid scoring, learned `expert_bias`, `norm_topk_prob=True`, `routed_scaling_factor=2.5`, `router_dtype=fp32` |
| Attention | GQA, 64 Q heads / 4 KV heads, `head_dim=64`, QK-norm, RoPE θ=8e6 |
| Context | 131072 |
| Vocab | 262144 |
| Declared languages | 23 |

**Parameter budget** — this is the number that determines what the experiment is actually about:

```
embeddings (untied, in+out) : 2.15B   ( 6.7%)
attention (19 layers)       : 0.68B   ( 2.1%)
layer-0 dense MLP           : 0.10B   ( 0.3%)
shared experts (18 layers)  : 0.23B   ( 0.7%)
routed experts (18 × 128)   : 28.99B  (90.2%)   ← the experiment
                              ------
                              32.15B
```

Active per token ≈ 2.4B, matching the spec. **Quantizing sarvam-30b is, to 90%, quantizing 2304
small expert matrices** (18 layers × 128 experts), each of which sees only the ~4.7% of tokens
routed to it.

### 1.3 Two facts about the code that shape everything

**(a) The router is a single, clean hook point.**

```python
# modeling_sarvam_moe.py:196-245
class SarvamMoEGate(nn.Module):
    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        scores = logits.sigmoid()
        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)
        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight, logits          # ← chosen experts, weights, full 128-dim logits
```

A single `register_forward_hook` on `model.model.layers[L].mlp.gate` yields **all three**
quantities RQ4 needs. No model surgery, no `output_router_logits` plumbing, no forked modeling
file. This is the entire RQ4 instrumentation surface, and it is why Phase 2 is tractable.

Caveat that drives a design decision: this works because the gate stays a live Python module.
It holds under HF `transformers` + `compressed-tensors` loading. It does **not** hold under vLLM,
which replaces the MoE block with a fused kernel. **All routing analysis therefore runs on the
HF path; vLLM is used only for throughput/latency measurement.**

**(b) Non-standard module names.** Attention is fused as `attention.query_key_value` and
`attention.dense` — not `q_proj/k_proj/v_proj/o_proj`. Experts are stored **unfused as 2D
`nn.Linear`** at `model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}`. Every
off-the-shelf quantization module map will silently miss these. Hence
[`quant/sarvam_moe_adapter.py`](src/indicquant/quant/sarvam_moe_adapter.py).

---

## 2. The logic: why this should happen

### 2.1 The generic mechanism (true of any model)

GPTQ, AWQ, and SmoothQuant all fit quantization parameters to **activation statistics** measured
on a calibration corpus. AWQ scales weight channels by observed activation magnitude to protect
salient channels; GPTQ uses the calibration Hessian to order and compensate weight updates.

If the calibration corpus is English (C4, WikiText, the Pile — near-universal practice) and the
deployment distribution is Tamil, the fitted parameters are optimal for a distribution the model
will not see. This is the spec's H1, and it applies to dense models too.

### 2.2 The MoE-specific mechanism (the sharper story)

`llm-compressor`'s own MoE tutorial states the problem the framework was built to solve:

> "Given that calibration datasets are relatively small, some experts will not be activated, or
> activated very infrequently. This can result in poorly calibrated quantization parameters,
> numerical instability, or NaNs."

Its fix is **linearization + `calibrate_all_experts`**: during calibration, route *all* tokens
through *all* experts, keeping only routed outputs for the forward result.

**This fixes the token-count problem. It does not fix the language problem.**

Under English calibration with `calibrate_all_experts=True`, an expert that specializes in Tamil
now receives plenty of activations — all of them English. Its scales are fit precisely to a
distribution it will never encounter at inference. The framework's own workaround thus *hides* a
count symptom while leaving the distribution pathology fully intact, and does so silently.

This gives a mechanism that is MoE-specific, falsifiable, and measurable:

```
English calibration corpus
        │
        ├─► routing-faithful: Indic-specialized experts receive ~0 tokens
        │                     → scales fit to noise or defaults
        │
        └─► calibrate_all_experts: Indic-specialized experts receive many tokens,
                                   all English → scales fit to the wrong distribution
        │
        ▼
  INT4 expert weights mis-scaled exactly where Indic capability lives
        │
        ├─► capability loss concentrated in low-resource languages   (RQ1, RQ2)
        └─► perturbed expert outputs → shifted router inputs at the
            next layer → different experts selected                 (RQ4)
```

The second branch is worth stating explicitly: the router at layer *L+1* consumes the output of
the MoE block at layer *L*. Quantization error in expert weights propagates into router **inputs**,
so routing can drift even when the gate matrix itself is never quantized. Distinguishing that
indirect path from direct gate quantization is what Condition I tests.

### 2.3 What this buys us that the spec doesn't have

1. **A near-free predictive pre-flight.** Expert coverage is measurable with **one FP16 forward
   pass and zero quantization**: hook the routers, run each calibration corpus, count tokens per
   expert. If English calibration starves experts that Indic text relies on, H1 is predicted
   before a single checkpoint is quantized. This de-risks the Phase 0 GO/NO-GO gate for a few
   GPU-hours instead of a few GPU-days.

2. **Condition H — routing-faithful vs. `calibrate_all_experts`.** The framework exposes this as
   a flag. The comparison is, as far as we can tell, unpublished, and it is directly actionable
   for anyone quantizing any MoE model.

3. **Condition I — quantize the gate vs. preserve it.** Separates direct router perturbation from
   the indirect upstream path in §2.2. The gate is 128×4096 per layer — **0.03% of parameters**.
   If preserving it in FP16 recovers routing agreement, that is a free, immediately deployable
   recommendation, and it is the kind of finding that lands in a systems venue.

---

## 3. Experimental design

### 3.1 Conditions

Conditions A–G are the spec's. H and I are added per §2.3.

| ID | Method | Bit-width | Calibration | Purpose |
|---|---|---|---|---|
| **A** | FP16 | 16 | — | Baseline. Every Δ is measured against this. |
| **B** | AWQ | INT4 | English (C4) | **Standard practice.** The thing we claim is wrong. |
| **C** | GPTQ | INT4 | English (C4) | Method generalization — is this AWQ-specific? |
| **D** | GGUF Q4_K_M | ~4 | English | Deployment reality check (llama.cpp path) |
| **E** | AWQ | INT4 | **Indic** | **B vs. E is the money comparison.** |
| **F** | AWQ | INT4 | **Mixed** (Indic + En + code-mixed) | H3: mixed beats either pure corpus |
| **G** | AWQ | INT8 | English | Bit-width ablation — does the gap widen as bits fall? |
| **H** | AWQ | INT4 | English + Indic | **New.** `calibrate_all_experts` on/off. Isolates coverage from distribution. |
| **I** | AWQ | INT4 | English + Indic | **New.** Gate quantized vs. FP16-preserved. Isolates direct router perturbation. |

**Held fixed across B/E/F/H/I:** method, bit-width, group size, sequence length, and **total
calibration token budget**. Only the calibration *distribution* varies. The token budget is
enforced in code ([`calib/build.py`](src/indicquant/calib/build.py)) and asserted in tests —
comparing distributions while accidentally varying volume would invalidate the headline result.

### 3.2 Languages

Stratified by resource level, not convenience ([`configs/languages.yaml`](configs/languages.yaml)):

| Tier | Languages | Scripts |
|---|---|---|
| High | Hindi, Bengali | Devanagari, Bengali |
| Medium | Tamil, Telugu, Marathi, Gujarati | Tamil, Telugu, Devanagari, Gujarati |
| Low | Odia, Assamese, Punjabi, Kannada | Odia, Bengali, Gurmukhi, Kannada |
| Code-mixed | Hinglish, Tanglish (romanized) | **Latin** |
| Control | English | Latin |

Marathi and Hindi share Devanagari; Assamese and Bengali share a script. That is deliberate — it
lets us separate *language* effects from *script* effects (RQ3) within the design rather than by
assertion.

### 3.3 Benchmarks

| Capability | Dataset | Why |
|---|---|---|
| Knowledge / reasoning (MCQ) | `ai4bharat/MILU` | Broadest native-Indic MCQ coverage |
| Translation (generation) | `google/IndicGenBench_flores_in` | Parallel — same content across languages |
| QA (generation) | `google/IndicGenBench_xquad_in`, `ai4bharat/IndicQA` | Extractive, gradeable |
| Summarization (generation) | `google/IndicGenBench_crosssum_in` | Long-form fluency |
| Instruction following | `ai4bharat/IndicIFEval` | Programmatically checkable |
| **Code-mixed** | **constructed** | No canonical benchmark exists — building one is a contribution |

MCQ accuracy is a weak proxy for the fluency loss quantization actually causes, so generation
tasks carry equal weight, and **script integrity** is measured directly
([`eval/script_integrity.py`](src/indicquant/eval/script_integrity.py)): fraction of output
codepoints in the expected Unicode block, malformed-grapheme rate, and silent drift to English.
A model that answers correctly in the wrong script has failed in a way accuracy will not show.

### 3.4 Metrics

**Capability** — per-language accuracy Δ from FP16; generation quality (chrF++ and native-script
metrics, avoiding English-tuned automatic metrics used uncritically); script integrity.

**Routing (RQ4)** — per-language expert-activation distribution; top-6 agreement with FP16;
router entropy shift; per-expert language affinity with a permutation null.

**Systems** — TTFT, tokens/sec, P95 latency, memory footprint, KV-cache pressure, and
**tokenizer fertility per language** (§7 Phase A — computable today, on a laptop).

**Rigor** — ≥5 seeds per configuration; bootstrap confidence intervals, never bare point
estimates; fixed and explicitly reported sampling parameters; a temperature ablation, because a
reviewer will ask whether the effect survives sampling changes.

---

## 4. System architecture

A research harness, not an application. Config-driven, artifact-cached, one CLI verb per stage.
Every stage writes a manifest recording inputs, config hash, and git SHA, so a stage is never
recomputed — re-quantizing a 32B model is the single dominant cost, and the pipeline is designed
around never doing it twice.

```
 configs/                    ┌──────────────────────────────────────────┐
   languages.yaml            │  Everything is addressed by condition ID │
   model/*.yaml       ──────►│  and language code. Artifacts are        │
   calibration/*.yaml        │  content-addressed by config hash.       │
   conditions/*.yaml         └──────────────────────────────────────────┘
        │
        ▼
 ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐   ┌──────────┐
 │ calib build │──►│   quantize   │──►│  evaluate   │──►│  analyze  │──►│  report  │
 │             │   │              │   │             │   │           │   │          │
 │ fixed token │   │ llm-compres. │   │ tasks +     │   │ bootstrap │   │ plots +  │
 │ budget      │   │ + sarvam     │   │ ROUTER      │   │ CIs,      │   │ tables + │
 │ per corpus  │   │   adapter    │   │ HOOKS       │   │ perm test │   │ model    │
 └─────────────┘   └──────────────┘   └─────────────┘   └───────────┘   │  cards   │
        │                  │                 │                │         └──────────┘
        ▼                  ▼                 ▼                ▼
   data/calib/       checkpoints/      results/*.parquet   results/figures/
```

### Design decisions worth stating

**Parquet, not JSON, for routing traces.** Phase 2 records, per token per layer, six expert IDs
plus six weights plus (subsampled) 128 fp16 logits. A 500-sequence × 512-token eval over 18
layers is ~4.6M rows before logits. Columnar storage with predicate pushdown makes
`metrics.py` queryable rather than a memory problem.

**Metrics never touch the model.** [`routing/metrics.py`](src/indicquant/routing/metrics.py)
consumes recorded Parquet only. Analysis is re-runnable, revisable, and reviewable without a GPU
— which matters when the GPU is rented by the hour.

**GPU code is quarantined, and split in two.** Core dependencies are CPU-installable so the
entire dev loop runs on a laptop; `llmcompressor`, `gptqmodel` and `vllm` are imported lazily
inside functions. They are further split into two *mutually exclusive* extras, because they
genuinely cannot coexist — verified by resolver failure on 2026-08-18:

> `llmcompressor>=0.13.0` requires `compressed-tensors>=0.18.0` and `transformers>=5.9.0`;
> every `vllm` release pins `compressed-tensors<=0.17.0` and `transformers<5`. uv's resolver
> reports them **incompatible**.

So the GPU node runs two virtualenvs — `.[quant]` for quantization and routing capture,
`.[serve]` for throughput and latency. `[tool.uv] conflicts` declares this so uv resolves them
separately instead of failing. The split also reinforces §6.3: routing capture happens on the
HF path, never under vLLM.

**The tiny model is the real model.** `configs/model/sarvam-30b-tiny.yaml` builds a random-init
`SarvamMoEConfig` (~5M params) that instantiates the **actual `SarvamMoEGate` class** via
`trust_remote_code`. Hooks, metrics, storage, and CLI plumbing are tested against real Sarvam
code paths on CPU, before the 128 GB download.

---

## 5. Module-by-module implementation

### `calib/` — calibration corpus construction

| File | Responsibility |
|---|---|
| `corpora.py` | Source definitions: `allenai/c4` (English), `ai4bharat/sangraha` (Indic), `ai4bharat/IndicCorpV2`, code-mixed construction |
| `build.py` | Assemble a corpus to an **exact token budget**, deterministic under seed, emit manifest |

The invariant: `build_corpus()` fills to a target *token* count (not document count) measured with
the sarvam tokenizer, and writes `n_tokens` to the manifest. `tests/test_calib_budget.py` asserts
all of B/E/F/H/I land within tolerance of each other. Without this, the headline comparison
confounds distribution with volume.

The code-mixed corpus is constructed, not downloaded: romanized Hindi/Tamil, drawn from
naturally-occurring romanized text where available and transliterated (`ai4bharat/Aksharantar`)
where not, with the mixing ratio recorded in the manifest.

### `quant/` — quantization backends

| File | Responsibility |
|---|---|
| `sarvam_moe_adapter.py` | **The piece that does not exist anywhere yet.** |
| `compressor.py` | `llm-compressor` oneshot driver: AWQ / GPTQ / INT8 |
| `gptq.py` | `GPTQModel` path — independent implementation, guards against single-library artifacts |
| `gguf.py` | llama.cpp conversion + Q4_K_M (Condition D) |

`sarvam_moe_adapter.py` provides three things:

1. **A module map** for `attention.query_key_value` / `attention.dense`, since no stock mapping
   matches these names.
2. **`SarvamMoELinearExperts`**, following `llm-compressor`'s `LinearExperts2D` contract
   (`__init__`, `from_experts_module`, `forward` gated on `get_calibrate_all_experts_flag`).
   Sarvam stores experts already-unfused as 2D linears, so this is a thin adapter rather than a
   3D→2D conversion — the easy case in the framework's taxonomy. This is the **Condition H**
   switch.
3. **A configurable `ignore` policy** over `gate`, `shared_experts`, and `lm_head`. This is the
   **Condition I** switch.

The file is written against `llm-compressor`'s documented extension contract so it can be
upstreamed into `llmcompressor/modeling/moe/sarvam_moe.py` essentially unchanged, plus an entry
in `conversion_mappings.py` and `test_linearize.py`. That PR is a headline deliverable: it is the
difference between "sarvam-30b cannot be AWQ-quantized" and "it can."

### `routing/` — the RQ4 instrumentation

**`hooks.py`** — `RouterRecorder`, a context manager:

```python
with RouterRecorder(model, out_dir=..., condition="B", language="ta") as rec:
    model(input_ids=batch)          # hooks fire; nothing else changes
# rec.flush() written on exit
```

Registers `register_forward_hook` on `model.model.layers[L].mlp.gate` for every
`L >= config.first_k_dense_replace`. The hook receives `(module, args, output)` and unpacks
`output` as `(topk_idx, topk_weight, logits)` directly — no interception of internals. Rows are
keyed `(condition, language, seq_id, position, layer)`. Top-6 IDs and weights are always stored;
full 128-dim logits are stored fp16 and position-subsampled at a configurable stride, because
storing them for every token is what turns a 4M-row table into a 500M-row one.

**`metrics.py`** — reads Parquet, never the model:

- `expert_coverage(trace)` → tokens per expert per layer. **The Phase 0 pre-flight.**
- `expert_activation_histogram(trace, language)` → 128-bin distribution per layer
- `topk_agreement(fp16_trace, quant_trace)` → overlap@6 and rank-weighted agreement.
  **Raises `RoutingComparisonError` unless the two traces share identical `input_ids`** (see §6.1).
- `router_entropy(trace)` → entropy over sigmoid-normalized 128-dim logits
- `language_affinity_matrix(trace)` → 128×12 per layer, with a token-shuffle permutation null so
  specialization is *tested*, not asserted

**`storage.py`** — Parquet schema, sharded writes, manifest handling.

### `eval/` — capability measurement

`tasks/` holds one adapter per benchmark, each exposing `load(language, n, seed)` and
`score(predictions, references)`. `runner.py` drives conditions × languages × seeds with
teacher-forced routing capture wired in. `script_integrity.py` implements the Unicode-block
checks from §3.3.

### `systems/` — the InferLite-strength measurements, done for real

`fertility.py` (tokens per word/character on parallel FLORES text — **runs today, on a laptop**),
`latency.py` (TTFT, tokens/sec, P95 via vLLM), `memory.py` (footprint, KV-cache pressure).

### `analysis/` — statistics and figures

`stats.py`: bootstrap CIs over seeds, paired tests for B-vs-E, permutation nulls for expert
specialization, and the resource-level trend fit that is the project's headline plot.
`plots.py`: the Phase 0 figure (Δ-accuracy vs. resource level, English- vs. Indic-calibrated), the
128×18 expert-activation heatmap, and the routing-agreement-by-language chart.

---

## 6. Methodological guards

Two mistakes would make Phase 2 unpublishable. Both are enforced in code rather than left to
discipline.

### 6.1 Teacher-forced routing comparison

Comparing FP16 and quantized routing on **free generation is meaningless**. The moment the two
models emit different tokens, the sequences diverge, and every subsequent routing difference
measures divergent context rather than quantization. The observed "routing drift" would be an
artifact, and a reviewer would catch it immediately.

All routing metrics are therefore computed on **identical, teacher-forced token sequences**: the
same `input_ids` are pushed through both models, and routing is compared position by position.
`topk_agreement()` raises `RoutingComparisonError` if the two traces' `input_ids` hashes differ.
`tests/test_metrics.py` asserts that it raises. This guard is the difference between a real
result and a retracted one.

### 6.2 The script confound

Indic scripts occupy disjoint Unicode blocks. An expert that appears "Tamil-specialized" may be
trivially "Tamil-codepoint-specialized" — a statement about the tokenizer, not the model, and not
interesting.

Three controls:

1. **Romanized code-mixed text is a control, not a curiosity.** Hinglish and Tanglish share the
   Latin script with English. Expert specialization that persists on romanized Tamil is
   specialization on *language*, not script. This is the discriminating test, and it is why H4
   sits at the center of the design rather than at the edge.
2. **Same-script language pairs.** Hindi/Marathi (Devanagari) and Bengali/Assamese (Bengali
   script) let us hold script constant while varying language.
3. **Depth stratification.** Script-level specialization should concentrate in early layers,
   semantic specialization in later ones. Reporting affinity per layer separates them.

### 6.3 Where routing is measured

Routing analysis runs on the HF `transformers` + `compressed-tensors` path, where
`SarvamMoEGate` remains a live Python module and hooks fire. vLLM fuses the MoE block and the
hooks would silently record nothing — so vLLM is used **only** for throughput and latency, never
for routing. Enforced by a runtime check in `RouterRecorder.__enter__`, which raises if it binds
zero hooks rather than producing an empty trace.

---

## 7. Execution phases

### Phase A — zero-GPU results (now, on the laptop)

Two deliverables need no GPU and no 128 GB download:

1. **Tokenizer fertility.** `tokenizer.json` alone is a few MB. Compute tokens-per-word and
   tokens-per-character across all 12 languages on **FLORES-200 parallel text** — same content in
   every language, so fertility is directly comparable. This produces a real plot and quantifies
   the compounding cost the spec argues for: Indic scripts yield more tokens per word, so equal
   prompts cost more, stress KV cache harder, and inflate every downstream latency number.
2. **Calibration corpora.** Build and freeze English / Indic / mixed / code-mixed at a fixed
   token budget, manifests committed.

```bash
uv run indicquant fertility --config configs/languages.yaml
uv run indicquant calib build --config configs/calibration/english_c4.yaml
```

### Phase B — literature check (blocking, before any spend)

The spec's Week-1 item, still open. Search Google Scholar, arXiv, and ACL Anthology for
*quantization × multilingual calibration*, *low-resource quantization degradation*, and *MoE
router quantization*. Record in [`docs/LITERATURE.md`](docs/LITERATURE.md). If RQ1–RQ3 are
already cleanly answered, pivot to RQ4 + Conditions H/I — narrower, and near-certainly open. The
scaffold supports that pivot without restructuring.

### Phase C — expert-coverage pre-flight (first GPU session, hours)

One FP16 load, one forward pass per calibration corpus, router hooks on. Yields tokens-per-expert
per corpus and the English-vs-Indic coverage delta. **Predicts the Phase 0 outcome before any
quantization runs.** The cheapest possible test of the core mechanism.

### Phase 0 — pilot, GO/NO-GO

Conditions A, B, E. Five languages: Hindi, Tamil, Odia, Hinglish, English. ≥5 seeds.

**Single deliverable: one plot** — Δ-accuracy vs. language resource level, English- vs.
Indic-calibrated. Gap visible → GO. No gap → **STOP**, publish the null result as a blog post.
Build nothing beyond what this plot requires.

### Phases 1–4

Full sweep (RQ1–RQ3) → router analysis (RQ4, the novel section) → Indic-calibrated checkpoints on
HF + the `llm-compressor` `sarvam_moe` PR → write-up. Detailed in the spec, §7.

---

## 8. Compute and cost

| Item | Figure |
|---|---|
| GPU | 1×H200 141GB (fp16 weights 64.3 GB, room for KV cache) or 2×H100 80GB |
| Disk | ~330 GB working set: 129 GB fp32 source + 64 GB bf16 + ~20 GB per INT4 checkpoint → provision 500 GB–1 TB NVMe |
| Phase 0 | ~30–40 GPU-hours ≈ **$200–300** including storage, egress, and false starts |

`scripts/cost_estimate.py` recomputes this from the active config, so the number is checked
rather than guessed, and re-checked whenever the condition or language set changes.

**Controls:** quantize once per condition and cache the checkpoint — never re-quantize. Cost the
pilot precisely before committing to Phase 1. **Cut languages before cutting seeds**: narrow scope
is forgivable, noisy claims are not.

---

## 9. Risks

| Risk | Handling |
|---|---|
| ~~`transformers` v5 vs. sarvam's remote code (`transformers_version: 4.57.2`)~~ | **RESOLVED 2026-08-18.** Sarvam's remote code loads and runs correctly under `transformers` 5.15.0 — verified by building the tiny model and running a forward pass (deprecation warnings, no errors). This was the risk flagged as the likeliest early blocker; it is not one. Note `llm-compressor` 0.13.0 pins `transformers<=5.14.1` while newer pre-releases require `>=5.15.0`, so the quant env's pin still needs care. |
| Custom-code MoE breaks `llm-compressor` graph tracing | `sarvam_moe_adapter.py` exists for exactly this; fall back to the `layer_sequential` pipeline |
| `llm-compressor` and `vllm` cannot share an environment | Confirmed by resolver failure. Two venvs on the GPU node, declared as conflicting extras. Discovering this live on a rented GPU would have cost a day. |
| No calibration-language gap (H1 false) | Phase 0 kill gate. One week lost, null result published — genuinely useful. |
| Prior work already covers RQ1–RQ3 | Phase B check; pivot to RQ4 + Conditions H/I |
| No language-specialized experts exist | H2 fails, RQ1–RQ3 stand. A clean negative MoE result is still novel and gets reported honestly. |
| Sarvam ships INT4 first | `sarvam-30b-fp8` and `-gguf` already exist, so the clock is real. The calibration-language *finding* survives regardless of who ships checkpoints. |
| Compute overrun | Cost the pilot; cut languages, not seeds |

---

## Appendix: provenance

Every architectural claim in §1 is traceable:

| Claim | Source |
|---|---|
| Params, layers, experts, top-k, router config | `sarvamai/sarvam-30b/config.json` |
| 128.6 GB, 26 shards, 7122 tensors | `model.safetensors.index.json` metadata |
| `SarvamMoEGate.forward` return signature | `modeling_sarvam_moe.py:236-245` |
| Fused attention / unfused expert naming | `model.safetensors.index.json` weight map |
| Tiny-config viability | `configuration_sarvam_moe.py` — all dims are plain kwargs |
| AutoAWQ archived | GitHub API, `casper-hansen/AutoAWQ`, `archived: true`, last push 2025-05-11 |
| llm-compressor MoE extension contract | `docs/developer-tutorials/add-moe-support.md`, `src/llmcompressor/modeling/moe/` |
| InferLite simulation mode | `llm-inferlite/backend/quantization/engine.py`, README |

Re-verify before Phase 0. Indic NLP and the quantization toolchain both moved fast through 2026.
