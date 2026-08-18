# Compute

`python scripts/cost_estimate.py --phase 0` recomputes everything below from the active
configs. Numbers here are the current output plus the reasoning behind the assumptions.

## Hardware

| | Requirement | Why |
|---|---|---|
| GPU | **1×H200 141GB**, or 2×H100 80GB | bf16 weights are 64.3 GB. One H100 80GB fits the weights but leaves little room for KV cache and activation buffers during quantization. |
| Disk | **500 GB–1 TB NVMe** | working set ~330 GB: 128.6 GB fp32 source + 64.3 GB bf16 + ~20 GB per INT4 checkpoint |
| Network | fast egress | 128.6 GB in, ~20 GB out per released checkpoint |

The activation sparsity is the whole point: 2.4B active of 32B total is what makes a 32B MoE
affordable on a single node. Quantization does not benefit from sparsity the way inference
does, though — every one of the 2304 expert matrices must be visited regardless of how rarely
it fires. Budget accordingly.

## Phase 0 estimate

```
GPU hours                  23.8 h   $ 83.42     (at $3.50/h)
storage (233 GB)                    $ 11.64
subtotal                            $ 95.06
+ 30% overhead                      $123.58
```

Breakdown: 1.5 h download, 2 h expert-coverage pre-flight, 12 h quantization (two conditions),
8.3 h evaluation (75,000 generations).

Call it **$150–250 in practice.** The estimate assumes nothing goes wrong, and something
always does — a failed quantization run, an OOM, a dataset config that changed name.

## The assumptions, and which one is shaky

| Assumption | Value | Confidence |
|---|---|---|
| quantization hours per condition | 6.0 | **low** — see below |
| download hours | 1.5 | medium (128.6 GB at ~250 MB/s) |
| eval generations/hour | 9000 | medium |
| pre-flight hours | 2.0 | medium |

**The quantization estimate is the shaky one.** AWQ on a dense 30B is a few hours. sarvam-30b
presents 2304 small expert matrices (18 layers × 128 experts) at `moe_intermediate_size=1024`
instead of a handful of large ones, and small GEMMs use a GPU badly. It could be 6 hours; it
could be 15. **Measure it on the first run and update `scripts/cost_estimate.py`** before
committing to Phase 1, where the same figure is multiplied by five conditions.

## Providers

Rent rather than chase free credits. Lambda, RunPod, and Vast all list H100/H200 on demand.
Indian providers (Yotta, E2E) may be cheaper and are a genuinely nice detail for a project
about Indic models — worth a price check before booking.

## Controls

- **Quantize once per condition, cache the checkpoint, never re-quantize.** Conditions H and
  I each contain two grid cells that are literally B and E; `expand_grid` marks them for
  reuse rather than re-running. This is the single most expensive avoidable mistake available.
- **Cut languages before cutting seeds.** Narrow scope is forgivable; noisy claims are not.
- Run the expert-coverage pre-flight before quantizing anything. It costs ~2 GPU-hours and
  can predict the Phase 0 outcome.
- Detach storage from the GPU instance between sessions if the provider allows it — storage
  is cheap, idle H200s are not.

## Environment setup on the node

`llm-compressor` and `vllm` **cannot share an environment** (conflicting `compressed-tensors`
and `transformers` pins — verified). Two venvs:

```bash
uv venv .venv-quant && uv pip install -e '.[quant]'   # quantize + routing capture
uv venv .venv-serve && uv pip install -e '.[serve]'   # throughput/latency only
```

Routing capture must run in the `quant` env: vLLM fuses the MoE block and gate hooks would
bind nothing (ARCHITECTURE.md §6.3).
