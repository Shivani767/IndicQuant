"""Model loading.

Two paths, one code path for hooks:

  - `load_model(...)`  — the real sarvam-30b, from Hugging Face, on a GPU node.
  - `build_tiny_model(...)` — a random-init model built from `SarvamMoEConfig` at ~5M params.

The tiny model is not a lookalike. `SarvamMoEConfig` accepts every dimension as a plain
constructor kwarg, so the tiny model instantiates the REAL `SarvamMoEGate` and
`SarvamMoESparseMoeBlock` classes from sarvam's remote code. Hooks, metrics, storage and CLI
plumbing are therefore exercised against genuine Sarvam code paths on a laptop, before the
128.6 GB download. Its weights are random, so nothing measured on it is a research result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from indicquant.config import ConfigError, load_model_config


def load_model(
    model_name: str = "sarvam-30b",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load a model + tokenizer for evaluation or routing capture.

    `checkpoint_path` loads a locally quantized checkpoint instead of the HF baseline.

    bfloat16, not float16: sarvam-30b ships fp32 on the Hub but was trained in bf16, and the
    router runs in fp32 (`router_dtype: fp32`). fp16 risks overflow on the way through.

    Routing capture requires this HF path. vLLM and llama.cpp fuse the MoE block, so gate
    hooks bind nothing there — see ARCHITECTURE.md §6.3.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_model_config(model_name)
    source = str(checkpoint_path) if checkpoint_path else cfg["hf_id"]
    if source is None:
        raise ConfigError(
            f"model config {model_name!r} has no hf_id. For control-dense this is expected: "
            "pin the model after verifying it on Hugging Face (see the config's TODO)."
        )

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["hf_id"], trust_remote_code=cfg.get("trust_remote_code", False)
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model.eval()
    return model, tokenizer


def build_tiny_model(model_name: str = "sarvam-30b-tiny", seed: int = 0) -> Any:
    """Build a random-init SarvamMoE model on CPU for testing.

    Downloads sarvam's remote *code* (a few small .py files) but none of its weights.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = load_model_config(model_name)
    if not cfg.get("random_init"):
        raise ConfigError(
            f"{model_name!r} is not a random-init config; use load_model() for real weights"
        )

    arch = cfg["architecture"]
    torch.manual_seed(seed)

    config = AutoConfig.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    for key, value in arch.items():
        if key in {"model_type", "class_name"}:
            continue
        setattr(config, key, value)
    # head_dim is derived in SarvamMoEConfig.__init__ when not passed, so set it explicitly
    # after the loop to make sure our value survives.
    if "head_dim" in arch:
        config.head_dim = arch["head_dim"]
    config.torch_dtype = "float32"

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def count_moe_layers(model_config: dict[str, Any]) -> list[int]:
    """MoE layer indices for a model config.

    Layers below `first_k_dense_replace` are dense and have no router. For sarvam-30b this
    is [1..18] — 18 MoE layers, not 19.
    """
    arch = model_config["architecture"]
    n_layers = arch["num_hidden_layers"]
    first_dense = arch.get("first_k_dense_replace", 0)
    return list(range(first_dense, n_layers))


def load_tokenizer(model_name: str = "sarvam-30b") -> Any:
    """Load just the tokenizer.

    This is the whole dependency for the fertility analysis: `tokenizer.json` is a few MB,
    against 128.6 GB for the weights. It is why Phase A produces a real result on a laptop.
    """
    from transformers import AutoTokenizer

    cfg = load_model_config(model_name)
    return AutoTokenizer.from_pretrained(
        cfg["hf_id"], trust_remote_code=cfg.get("trust_remote_code", False)
    )
