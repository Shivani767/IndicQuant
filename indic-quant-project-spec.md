# IndicQuant

### Calibration-Language Mismatch and Expert-Routing Drift in Quantized Indic MoE Models

**Full project specification — Shivani Bhandari**

Target companies: Sarvam AI, Krutrim, Microsoft (AI Frameworks / ONNX Runtime / Olive), BharatGen

---

## 1. Thesis

Every mainstream post-training quantization method — GPTQ, AWQ, SmoothQuant — estimates activation statistics from a calibration corpus. That corpus is, in near-universal practice, English: C4, WikiText, the Pile.

India's sovereign LLMs are not English models. Sarvam-30B and Sarvam-105B allocate a substantial share of their training budget to Indic languages and code-mixed text.

**Claim: quantizing an Indic model with English calibration data degrades Indic capability disproportionately — and the degradation is largest in exactly the low-resource languages these models exist to serve.**

If true, every practitioner running INT4 Sarvam today is silently discarding the model's reason for existing. The fix is free: change the calibration distribution, keep the bit-width.

Second, MoE-specific claim: with only ~2.4B of 32B parameters active per token, quantization error may perturb the **router**, not just the weights. If experts specialize by language or script, quantization could cause the model to select the wrong expert for Tamil input. This is a failure mode with no analogue in dense models, and it is essentially unstudied.

---

## 2. Why this project, for you

| Requirement | Your position |
|---|---|
| Quantization internals (GPTQ/AWQ/GGUF) | Deepest skill, proven across three independent contexts |
| Benchmarking harness | InferLite already does runtime + quant sweeps; needs per-language extension |
| ONNX / deployment path | On your resume three times; the Microsoft bridge |
| Reading Indic model outputs | You can qualitatively evaluate Hindi/Punjabi output. A Bay Area researcher structurally cannot. |
| Access | Models are Apache 2.0 on Hugging Face. No gatekeeping. |
| Compute feasibility | 32B MoE with 2.4B active quantizes on rented single-node H100 |

This is the first project where the artifact is *directly useful to the company you are targeting*. You would be handing Sarvam working quantized checkpoints of their own model plus an explanation of why the standard recipe fails.

---

## 3. Prior art and the gap

**What exists:** extensive quantization literature on English dense models; multilingual evaluation literature; MoE architecture papers.

**What does not exist (verify this before starting):** systematic measurement of calibration-language mismatch, and any study of quantization-induced routing drift in language-specialized MoE models.

**Action item — Week 1:** search Google Scholar, arXiv, and ACL Anthology for *quantization multilingual calibration*, *low-resource quantization degradation*, and *MoE router quantization*. If a paper has already done this cleanly, pivot to the MoE-routing angle alone, which is narrower and almost certainly open.

---

## 4. Research questions

**RQ1 — Language gap.** Does English-calibrated quantization degrade Indic-language performance more than English performance, at equal bit-width?

**RQ2 — Resource stratification.** Does the degradation scale inversely with a language's share of pretraining data? (Hindi should suffer least among Indic; low-resource scheduled languages most.)

**RQ3 — Script and tokenizer effects.** Do Devanagari, Dravidian scripts, and romanized/code-mixed input degrade differently? Tokenizer fertility differs sharply across these and may confound or amplify the effect.

**RQ4 — Router drift (the novel one).** Does quantization shift the expert-activation distribution? Do inputs that previously routed to a language-specialized expert now route elsewhere?

**RQ5 — The fix.** Does Indic-calibrated quantization recover capability at identical bit-width and zero inference cost? How much? Does it transfer across languages not in the calibration set?

---

## 5. Hypotheses

**H1 (primary).** Degradation is inversely proportional to a language's pretraining representation. Plotted as *degradation vs. resource level*, the curve slopes down and steepens as bit-width falls.

**H2 (MoE).** Router entropy increases and top-k expert agreement with FP16 falls under quantization, disproportionately for low-resource languages.

**H3 (fix).** Indic-calibrated quantization recovers a substantial fraction of the gap. A mixed corpus (Indic + English + code-mixed) outperforms either pure corpus, because the model must serve both.

**H4 (code-mixing).** Romanized Hinglish/Tanglish degrades worst of all — it is underrepresented in both English and formal-Indic calibration sets, yet it is how Indians actually type.

H4 is the finding most likely to be practically important and least likely to have been measured.

---

## 6. Experimental design

### Models
- **Sarvam-30B** — primary. MoE, ~32B total / ~2.4B active, GQA, 65K context.
- **Sarvam-105B** — stretch, if compute allows. MLA attention; different quantization sensitivity profile, which is itself interesting.
- **A dense multilingual control** (e.g. a Llama or Qwen multilingual variant) — essential for isolating MoE-specific effects from generic multilingual effects. Without this control, RQ4 is unfalsifiable.

*Verify current model cards on Hugging Face before committing — architecture details below come from secondary sources.*

### Quantization conditions
| Condition | Method | Calibration |
|---|---|---|
| A | FP16 | — (baseline) |
| B | AWQ-INT4 | English (C4) — *standard practice* |
| C | GPTQ-INT4 | English (C4) |
| D | GGUF Q4_K_M | English |
| E | AWQ-INT4 | **Indic** |
| F | AWQ-INT4 | **Mixed (Indic + English + code-mixed)** |
| G | AWQ-INT8 | English — bit-width ablation |

B vs. E is the money comparison. Hold method and bit-width fixed; vary only calibration distribution.

### Languages
Stratify deliberately by resource level, not convenience:
- **High:** Hindi, Bengali
- **Medium:** Tamil, Telugu, Marathi, Gujarati
- **Low:** Odia, Assamese, Punjabi, Kannada
- **Code-mixed:** romanized Hinglish, Tanglish
- **Control:** English

Roughly 10–12 languages. Enough for a resource-level trend line; not so many that per-language sample sizes collapse.

### Benchmarks
- **IndicBench**, **IndQA**, and current Indic evaluation suites — *verify what is standard now; this space moved fast through 2026.*
- **Generation quality** on Indic tasks, not just multiple-choice. MCQ accuracy is a poor proxy for the fluency loss quantization actually causes.
- **A code-mixed evaluation set** — you may need to construct this. If it doesn't exist, building it is a contribution in itself.

### Metrics
**Capability**
- per-language task accuracy, Δ from FP16
- generation quality (Indic-appropriate metrics; avoid English-tuned automatic metrics uncritically)
- script integrity — does it emit malformed Devanagari, wrong-script tokens, or drift to English?

**MoE-specific (RQ4)**
- expert-activation distribution per language, FP16 vs. quantized
- top-k routing agreement with FP16
- router entropy shift
- per-expert language affinity: do language-specialized experts exist at all?

**Systems (your existing InferLite strengths)**
- TTFT, tokens/sec, P95 latency
- memory footprint, KV cache pressure
- **tokenizer fertility per language** — Indic scripts produce more tokens per word, so equal-length prompts cost more and stress KV cache harder. Quantify it; it compounds every other cost.

### Rigor
- ≥5 seeds per configuration; report confidence intervals, never bare point estimates
- fixed sampling parameters across conditions, reported explicitly
- temperature ablation — quantization interacts with sampling and a reviewer will ask
- calibration set size held constant across B/E/F so you compare distribution, not volume
- release harness, calibration sets, configs, and raw outputs

---

## 7. Phased plan

### Phase 0 — Pilot (1 week) · **GO / NO-GO**

Sarvam-30B. Conditions A, B, E only. Five languages: Hindi, Tamil, Odia, Hinglish, English.

**Single deliverable: one plot — Δ-accuracy vs. language resource level, English-calibrated vs. Indic-calibrated.**

- **Gap visible →** GO.
- **No gap →** STOP. Publish the null result as a blog post ("quantization calibration language doesn't matter, and here's the evidence") — genuinely useful, and it cost you a week.

Build nothing beyond what this plot requires.

### Phase 1 — Full language sweep (4 weeks)
All conditions × all languages. Answers RQ1–RQ3. Sufficient alone for a workshop paper and a strong blog post.

### Phase 2 — Router analysis (3 weeks)
Instrument expert activations; compare FP16 vs. quantized routing. Answers RQ4. **This is the section that makes it novel rather than merely useful.** Requires hooking the MoE layer — real engineering, and a good story in interviews.

### Phase 3 — Indic calibration + release (4 weeks)
Optimize the calibration mixture, run ablations, ship quantized checkpoints to Hugging Face with a proper model card and reproducible recipe.

### Phase 4 — Write-up and outreach (2 weeks)

Total: ~14 weeks. Blog-post-worthy output at the end of Phase 1 (week 5).

---

## 8. Compute budget

Rent rather than chase free credits — Lambda, RunPod, Vast, or an Indian provider (Yotta, E2E) which may be cheaper and is a nice narrative detail.

**Cost drivers:** calibration passes (short), evaluation rollouts (the real cost — languages × conditions × seeds × benchmark size), and quantized-checkpoint storage.

**Controls:**
- 30B-with-2.4B-active is the whole point — activation sparsity makes this affordable
- quantize once per condition, cache checkpoints, never re-quantize
- Phase 0 on a single H100 for a day or two
- **cut languages before cutting seeds.** Narrow scope is forgivable; noisy claims are not.

Cost the pilot precisely, then extrapolate before committing to Phase 1.

---

## 9. Deliverables

| Artifact | Audience |
|---|---|
| **Indic-calibrated quantized Sarvam checkpoints on HF** | Sarvam, Indian developer community. The highest-signal item. |
| Open-source evaluation harness (InferLite extension) | Recruiters, OSS community |
| Blog post with the resource-level degradation plot | Everyone; your public stake in the ground |
| Workshop / conference paper | Research credibility |
| Upstream PRs: Indic calibration support in AutoAWQ / GPTQModel / **Olive** | Microsoft bridge — a concrete OSS contribution in your specialty |

The Olive PR is how this single project serves both target companies at once.

---

## 10. Venue and career targeting

**Sarvam.** Ship checkpoints and the blog post first, then reach out with the artifact in hand. Do not apply cold and mention a project idea — send them working quantized versions of their own model. That is a fundamentally different conversation.

**Microsoft.** Route the same work through ONNX Runtime and Olive: export quantized Sarvam to ONNX, add a multilingual calibration path, contribute upstream. Multilingual edge deployment sits squarely in AI Frameworks' remit, and India-market relevance helps.

**Publication ladder**
1. Blog post after Phase 1 — immediate, no gatekeeping
2. Efficient-ML or multilingual-NLP **workshop** (NeurIPS/ICML/ACL) after Phase 2
3. **MLSys** or **NeurIPS Datasets & Benchmarks** after Phase 3
4. ACL/EMNLP main track if the router-drift finding is strong and clean

Do not aim only at a top-tier main track. The blog post and the HF checkpoints are what get you hired; the paper is a bonus.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| No calibration-language gap (H1 false) | Phase 0 kill gate — one week lost, null result published |
| Prior work already covers this | Week-1 literature check; fall back to the router-drift angle |
| Benchmarks unavailable or unstable | Verify suites before Phase 0; be ready to build a code-mixed eval set |
| No language-specialized experts exist | H2 fails but RQ1–RQ3 stand; report the negative MoE result honestly — it is still novel |
| Compute overrun | Cost the pilot; cut languages, not seeds |
| Model card details differ from assumptions | Verify architecture on HF before designing hooks |
| Sarvam releases their own quantized versions first | Ship fast; your calibration-mismatch *finding* survives regardless of who ships checkpoints |

---

## 12. Week 1 checklist

1. Pull Sarvam-30B from Hugging Face; confirm architecture, tokenizer, licence, and MoE layer structure
2. Literature check on quantization × multilingual calibration and MoE router quantization
3. Confirm current Indic benchmark standards (IndicBench, IndQA, alternatives)
4. Assemble three calibration corpora: English (C4 subset), Indic, code-mixed
5. Extend InferLite with per-language metric tracking
6. Price the Phase 0 run

Then run the pilot and look at one plot.

---

## Note on sourcing

Architecture details (parameter counts, active parameters, attention type, context windows) come from secondary reporting, not primary model cards. **Verify everything on Hugging Face before designing experiments around it.** Benchmark standards in Indic NLP moved quickly through 2026 and may have shifted again.
