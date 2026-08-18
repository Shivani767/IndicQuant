"""Memory footprint and KV-cache pressure.

KV-cache size is computable in closed form from the architecture, so the per-language
projection needs no GPU — only measured fertility. That makes it a Phase A deliverable
alongside `fertility.py`, and it turns the spec's qualitative claim ("Indic scripts stress KV
cache harder") into a number.
"""

from __future__ import annotations

from typing import Any


def kv_cache_bytes_per_token(model_config: dict[str, Any], dtype_bytes: int = 2) -> int:
    """KV-cache bytes per token.

    2 (K and V) x num_key_value_heads x head_dim x num_layers x dtype_bytes.

    Sarvam-30b's GQA ratio is aggressive — 64 query heads against 4 KV heads — so the cache
    is 16x smaller than MHA would give. That is what makes a 131072 context tractable, and
    it is worth reporting because it changes which of the spec's cost claims actually bind.
    """
    arch = model_config["architecture"]
    return (
        2 * arch["num_key_value_heads"] * arch["head_dim"] * arch["num_hidden_layers"] * dtype_bytes
    )


def kv_cache_gb(
    model_config: dict[str, Any],
    n_tokens: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    per_token = kv_cache_bytes_per_token(model_config, dtype_bytes)
    return per_token * n_tokens * batch_size / 1e9


def weight_memory_gb(model_config: dict[str, Any], bits: int) -> float:
    """Weight memory at a given bit-width.

    Embeddings and the LM head stay at bf16 (they are in every condition's `ignore` list),
    so the saving applies only to the quantized remainder. For sarvam-30b that remainder is
    90% of parameters — the routed experts — which is why INT4 is worth doing at all.
    """
    arch = model_config["architecture"]
    total = arch["total_parameters"]
    embed_params = 2 * arch["vocab_size"] * arch["hidden_size"]  # untied in + out
    quantized = total - embed_params

    if bits == 16:
        return total * 2 / 1e9
    # 4-bit group-128 carries a scale and zero-point per group: ~0.25 extra bits/weight.
    bits_per_weight = bits + (0.25 if bits == 4 else 0.125)
    return (quantized * bits_per_weight / 8 + embed_params * 2) / 1e9


def projected_language_cost(
    model_config: dict[str, Any],
    tokens_per_word: float,
    prompt_words: int = 200,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Project per-language KV-cache cost for an equal-content prompt.

    Takes measured fertility and returns what a 200-word prompt actually costs in that
    language. This is the concrete form of "fertility compounds every other cost".
    """
    n_tokens = int(round(prompt_words * tokens_per_word))
    return {
        "prompt_words": prompt_words,
        "prompt_tokens": n_tokens,
        "kv_cache_mb": kv_cache_gb(model_config, n_tokens, dtype_bytes=dtype_bytes) * 1000,
    }
