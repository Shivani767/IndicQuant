# Literature check

**Status: NOT DONE. This blocks GPU spend.**

The arXiv API returned empty responses from the environment where this repo was scaffolded,
so the search must be re-run from an unrestricted network. This is the spec's Week-1 item and
the single outstanding task before Phase 0.

## Why it blocks

If someone has already measured calibration-language mismatch cleanly, RQ1–RQ3 are answered
and the project should pivot to RQ4 plus Conditions H and I — narrower, and near-certainly
open. Finding that out **after** spending $200–300 and three weeks would be an avoidable
loss. The scaffold supports the pivot without restructuring.

## Searches to run

Google Scholar, arXiv, ACL Anthology, and Semantic Scholar:

**Calibration-language mismatch (RQ1–RQ3)**
- `quantization multilingual calibration`
- `low-resource language quantization degradation`
- `calibration data quantization language`
- `post-training quantization multilingual LLM`
- `GPTQ AWQ calibration dataset choice`
- `quantization Indic languages`

**MoE router quantization (RQ4)**
- `mixture of experts quantization router`
- `MoE expert routing quantization drift`
- `expert specialization language mixture of experts`
- `quantization router logits perturbation`
- `MoE calibration expert coverage`

**Conditions H and I specifically**
- `calibrate all experts quantization MoE`
- `router gate precision mixture of experts`

## What counts as a hit

| Finding | Consequence |
|---|---|
| A paper cleanly measuring calibration-language mismatch on multilingual models | RQ1–RQ3 are answered. Pivot to RQ4 + H/I. Cite and build on it. |
| A paper measuring it on **dense** models only | RQ1–RQ3 still stand for MoE, and the contrast becomes part of the story. Proceed. |
| A paper on MoE router quantization | Read carefully. If it covers language-specialized routing, the project's novel claim is gone and RQ5 (the fix) carries it. |
| Nothing on either | Proceed as designed. |
| Anything on Indic-specific quantization | Cite regardless; likely useful for benchmark selection. |

## Record findings below

Format: citation, one-line summary, and — most importantly — **what it does and does not
cover**, since the gap is what matters here.

<!-- entries go here -->

_(empty — search not yet run)_
