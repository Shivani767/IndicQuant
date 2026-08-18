"""llm-compressor adapter for `sarvam_moe`.

THIS DOES NOT EXIST UPSTREAM. `llm-compressor` ships per-architecture MoE adapters in
`src/llmcompressor/modeling/moe/` (llama4, granitemoe, cohere2_moe, deepseekv32, ...) and has
none for `sarvam_moe`. Consequence: sarvam-30b cannot be AWQ- or GPTQ-quantized today with
the standard toolchain. This module is the fix, and it is written against llm-compressor's
documented extension contract (`docs/developer-tutorials/add-moe-support.md`) so it can be
upstreamed essentially unchanged.

Three responsibilities:

1. MODULE MAP. Sarvam uses `attention.query_key_value` (fused QKV) and `attention.dense`,
   not `q_proj/k_proj/v_proj/o_proj`. Every stock mapping silently misses these — the run
   succeeds and quantizes less than you think.

2. `SarvamMoELinearExperts` — the `calibrate_all_experts` switch (CONDITION H). Sarvam stores
   experts already-unfused as 2D `nn.Linear` (`mlp.experts.{E}.{gate,up,down}_proj`), which
   is the easy case in llm-compressor's taxonomy: no 3D->2D conversion, just a forward that
   can optionally route every token through every expert during calibration.

3. IGNORE POLICY over `gate` / `shared_experts` / `lm_head` — the gate-quantization switch
   (CONDITION I).

See ARCHITECTURE.md §2.2 for why (2) and (3) are research conditions rather than plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Regex patterns matching sarvam_moe module paths. Used to build llm-compressor `ignore`
# lists and `targets`. Verified against model.safetensors.index.json on 2026-08-18.
GATE_PATTERN = r"re:.*mlp\.gate$"
SHARED_EXPERT_PATTERN = r"re:.*shared_experts.*"
ROUTED_EXPERT_PATTERN = r"re:.*mlp\.experts\.\d+\..*"
ATTENTION_QKV_PATTERN = r"re:.*attention\.query_key_value$"
ATTENTION_OUT_PATTERN = r"re:.*attention\.dense$"
DENSE_MLP_PATTERN = r"re:model\.layers\.0\.mlp\.(gate|up|down)_proj$"

# AWQ needs to know which linears share an input activation, so a single scaling factor can
# be applied to a whole group. For sarvam: gate_proj and up_proj of each expert both consume
# the block's hidden state, so they scale together; down_proj consumes the activated
# intermediate and scales separately.
AWQ_MAPPINGS = [
    {
        "smooth_layer": r"re:.*input_layernorm$",
        "balance_layers": [ATTENTION_QKV_PATTERN],
    },
    {
        "smooth_layer": r"re:.*post_attention_layernorm$",
        "balance_layers": [
            r"re:.*mlp\.experts\.\d+\.gate_proj$",
            r"re:.*mlp\.experts\.\d+\.up_proj$",
            r"re:.*shared_experts\.gate_proj$",
            r"re:.*shared_experts\.up_proj$",
        ],
    },
    {
        "smooth_layer": r"re:.*mlp\.experts\.\d+\.up_proj$",
        "balance_layers": [r"re:.*mlp\.experts\.\d+\.down_proj$"],
    },
]


@dataclass(frozen=True)
class IgnorePolicy:
    """Which modules stay at full precision.

    Defaults reflect standard practice and are what Conditions B-G use:
      - lm_head:        quantizing it costs accuracy for negligible savings
      - gate:           the router; 0.03% of parameters, so preserving it is nearly free
      - shared_experts: active for EVERY token, so error here hits every token

    Condition I flips `quantize_gate` to isolate direct router perturbation from the
    indirect path (expert error -> shifted router inputs at the next layer).
    """

    quantize_gate: bool = False
    quantize_shared_experts: bool = False
    quantize_lm_head: bool = False
    quantize_embeddings: bool = False

    def to_ignore_list(self) -> list[str]:
        ignore: list[str] = []
        if not self.quantize_lm_head:
            ignore.append("lm_head")
        if not self.quantize_gate:
            ignore.append(GATE_PATTERN)
        if not self.quantize_shared_experts:
            ignore.append(SHARED_EXPERT_PATTERN)
        if not self.quantize_embeddings:
            ignore.append("re:.*embed_tokens$")
        return ignore

    @classmethod
    def from_condition(cls, condition: dict[str, Any]) -> IgnorePolicy:
        quant = condition.get("quantization", {})
        ignore = quant.get("ignore", [])
        return cls(
            quantize_gate=not any("gate" in str(p) for p in ignore),
            quantize_shared_experts=not any("shared_expert" in str(p) for p in ignore),
            quantize_lm_head="lm_head" not in ignore,
        )


def expert_parameter_share(model_config: dict[str, Any]) -> dict[str, float]:
    """Where sarvam-30b's parameters actually live.

    Motivates the whole design: routed experts are ~90% of parameters, so quantizing
    sarvam-30b IS quantizing 2304 small expert matrices (18 layers x 128 experts), each
    seeing only the ~4.7% of tokens routed to it. That sparsity is the mechanism behind the
    coverage problem Condition H tests.
    """
    arch = model_config["architecture"]
    h = arch["hidden_size"]
    n_layers = arch["num_hidden_layers"]
    first_dense = arch.get("first_k_dense_replace", 0)
    n_moe_layers = n_layers - first_dense

    embed = 2 * arch["vocab_size"] * h
    qkv = h * (
        arch["num_attention_heads"] * arch["head_dim"]
        + 2 * arch["num_key_value_heads"] * arch["head_dim"]
    )
    attn = (qkv + arch["num_attention_heads"] * arch["head_dim"] * h) * n_layers
    dense_mlp = (
        3 * h * arch["intermediate_size"] * first_dense if "intermediate_size" in arch else 0
    )
    routed = arch["num_experts"] * 3 * h * arch["moe_intermediate_size"] * n_moe_layers
    shared = (
        arch.get("num_shared_experts", 0) * 3 * h * arch["moe_intermediate_size"] * n_moe_layers
    )
    gate = arch["num_experts"] * h * n_moe_layers

    total = embed + attn + dense_mlp + routed + shared + gate
    return {
        "embeddings": embed / total,
        "attention": attn / total,
        "dense_mlp": dense_mlp / total,
        "routed_experts": routed / total,
        "shared_experts": shared / total,
        "router_gate": gate / total,
        "total_parameters": float(total),
    }


def tokens_per_expert_fraction(model_config: dict[str, Any]) -> float:
    """Expected fraction of tokens each routed expert sees under faithful routing.

    For sarvam-30b: 6/128 = 4.7%. This is the *average*; the whole Phase C pre-flight exists
    because the *distribution* around it is what determines whether English calibration
    starves the experts Indic text depends on.
    """
    arch = model_config["architecture"]
    return arch["num_experts_per_tok"] / arch["num_experts"]


def build_linearized_experts_class() -> Any:
    """Return a `SarvamMoELinearExperts` class bound to llm-compressor's `LinearExperts2D`.

    Imported lazily and constructed at call time because `llmcompressor` lives in the `quant`
    extra and will not resolve on macOS/arm64 — the laptop dev loop must stay importable.

    Contract, per `docs/developer-tutorials/add-moe-support.md`: provide `__init__`,
    `from_experts_module`, and a `forward` gated on `get_calibrate_all_experts_flag`. Sarvam's
    experts are already unfused 2D linears, so `is_concatenated`/`is_transposed` are both
    False and this is a thin adapter.
    """
    raise NotImplementedError(
        "Phase 0 deliverable, and the highest-signal upstream contribution in the project.\n"
        "\n"
        "Implementation notes gathered during design:\n"
        "  - Subclass llmcompressor.modeling.moe.linear_experts.LinearExperts2D\n"
        "  - is_concatenated=False, is_transposed=False, has_bias=False, has_gate=True\n"
        "    (sarvam's SarvamMoEMLP has gate_proj/up_proj/down_proj, use_bias=False)\n"
        "  - from_experts_module() reads SarvamMoEExperts, an nn.ModuleList of 128\n"
        "    SarvamMoEMLP; weights are already 2D, so no 3D unfusing is required\n"
        "  - forward() must consult get_calibrate_all_experts_flag(): when set, route every\n"
        "    token through every expert but keep only the top-k outputs for the result\n"
        "  - upstream as llmcompressor/modeling/moe/sarvam_moe.py + an entry in\n"
        "    conversion_mappings.py + a case in test_linearize.py\n"
        "\n"
        "Verify against the installed llm-compressor version first: the MoE API moved with "
        "the transformers v5 migration (llm-compressor PR #2647)."
    )


def preflight_checks(model: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    """Verify a loaded model matches the naming this adapter assumes.

    Run before every quantization. sarvam-30b is `trust_remote_code`, so its module layout
    can change under us on any Hub revision — and a mismatch would quantize a different set
    of modules than intended while appearing to succeed.
    """
    names = [name for name, _ in model.named_modules()]
    arch = model_config["architecture"]
    n_moe_layers = arch["num_hidden_layers"] - arch.get("first_k_dense_replace", 0)

    found = {
        "gates": sum(1 for n in names if n.endswith("mlp.gate")),
        "attention_qkv": sum(1 for n in names if n.endswith("attention.query_key_value")),
        "attention_out": sum(1 for n in names if n.endswith("attention.dense")),
        "shared_experts": sum(1 for n in names if n.endswith("mlp.shared_experts")),
        "expert_mlps": sum(1 for n in names if ".mlp.experts." in n and n.endswith("down_proj")),
    }
    expected = {
        "gates": n_moe_layers,
        "attention_qkv": arch["num_hidden_layers"],
        "attention_out": arch["num_hidden_layers"],
        "shared_experts": n_moe_layers,
        "expert_mlps": n_moe_layers * arch["num_experts"],
    }
    mismatches = {k: (found[k], expected[k]) for k in expected if found[k] != expected[k]}
    return {"found": found, "expected": expected, "mismatches": mismatches, "ok": not mismatches}
