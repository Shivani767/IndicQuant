"""Config integrity.

These catch the class of mistake that is expensive to find on a rented GPU: a condition that
silently differs from its sibling in more than the one dimension it is supposed to vary.
"""

from __future__ import annotations

import pytest

from indicquant.config import (
    list_conditions,
    load_condition_config,
    load_languages,
    load_model_config,
)


def test_all_conditions_load():
    ids = [load_condition_config(stem.split("_")[0])["id"] for stem in list_conditions()]
    assert set(ids) == set("ABCDEFGHI")


def test_b_and_e_differ_only_in_calibration():
    """THE money comparison. If these diverge in any other field, the result is confounded."""
    b = load_condition_config("B")
    e = load_condition_config("E")

    assert b["calibration"] != e["calibration"]
    assert b["method"] == e["method"]
    assert b["bits"] == e["bits"]
    assert b["model"] == e["model"]

    bq, eq = b["quantization"], e["quantization"]
    for key in ("scheme", "group_size", "symmetric", "targets", "ignore", "calibrate_all_experts"):
        assert bq[key] == eq[key], f"B and E differ in quantization.{key}: {bq[key]} vs {eq[key]}"


def test_gate_and_shared_experts_are_preserved_by_default():
    """Conditions B-G must preserve the router, so any routing drift they show is the
    indirect upstream path. Condition I is the one that flips this."""
    for cid in "BCEFG":
        cfg = load_condition_config(cid)
        ignore = cfg["quantization"]["ignore"]
        assert any("gate" in str(p) for p in ignore), f"condition {cid} quantizes the router"
        assert any("shared_expert" in str(p) for p in ignore), (
            f"condition {cid} quantizes shared experts"
        )


def test_condition_i_has_a_gate_quantized_variant():
    cfg = load_condition_config("I")
    variants = cfg["ignore_variants"]
    assert any("gate" in str(p) for p in variants["gate_preserved"])
    assert not any("mlp.gate" in str(p) for p in variants["gate_quantized"])


def test_grid_conditions_reuse_existing_checkpoints():
    """H and I each contain cells that are literally B and E. Re-quantizing a 32B model
    twice for no reason is the most expensive avoidable mistake in the project."""
    from indicquant.quant.compressor import expand_grid

    for cid in ("H", "I"):
        cfg = load_condition_config(cid)
        cells = expand_grid(cfg)
        assert len(cells) == 4
        reused = [c for c in cells if "_reuse_condition" in c]
        assert len(reused) == 2, f"condition {cid} should reuse 2 of 4 cells, got {len(reused)}"


def test_condition_a_is_not_quantizable():
    from indicquant.quant.compressor import QuantizationError, build_plan

    with pytest.raises(QuantizationError, match="FP16 baseline"):
        build_plan(load_condition_config("A"))


def test_gguf_condition_declares_routing_unavailable():
    """llama.cpp fuses the MoE block. Recording that explicitly stops an absent trace from
    later being read as an absence of drift."""
    cfg = load_condition_config("D")
    assert cfg["capture_routing"] is False
    assert cfg.get("routing_unavailable_reason")


def test_plan_resolves_without_a_gpu():
    from indicquant.quant.compressor import build_plan

    plan = build_plan(load_condition_config("B"))
    assert plan.method == "awq"
    assert plan.scheme == "W4A16"
    assert plan.calibration == "english_c4"
    assert "lm_head" in plan.ignore
    assert plan.describe()


def test_plans_for_b_and_e_differ_only_in_calibration_and_output():
    from indicquant.quant.compressor import build_plan

    b = build_plan(load_condition_config("B"))
    e = build_plan(load_condition_config("E"))
    assert b.calibration != e.calibration
    assert b.output_dir != e.output_dir
    assert (b.method, b.scheme, b.group_size, b.symmetric, b.ignore, b.calibrate_all_experts) == (
        e.method,
        e.scheme,
        e.group_size,
        e.symmetric,
        e.ignore,
        e.calibrate_all_experts,
    )


# -- languages --------------------------------------------------------------------------


def test_language_set_spans_every_tier():
    langset = load_languages()
    tiers = {lang.tier for lang in langset}
    assert tiers == {"control", "high", "medium", "low", "code_mixed"}
    assert len(langset) >= 12


def test_phase0_subset_spans_the_resource_range():
    """The pilot must contain a high, a medium/low, a code-mixed and the control, or the
    GO/NO-GO plot has no trend to show."""
    langset = load_languages()
    tiers = {lang.tier for lang in langset.phase0()}
    assert "control" in tiers
    assert "code_mixed" in tiers
    assert len(tiers) >= 4


def test_script_controls_exist():
    """Same-script language pairs are what separate script effects from language effects."""
    groups = load_languages().script_groups()
    multi = {s: [lang.code for lang in ls] for s, ls in groups.items() if len(ls) > 1}
    assert "Devanagari" in multi and {"hi", "mr"} <= set(multi["Devanagari"])
    assert "Bengali" in multi and {"bn", "as"} <= set(multi["Bengali"])


def test_code_mixed_languages_use_latin_script():
    """The whole point: romanized Indic shares a script with English, so specialization
    that survives there is about language, not codepoints."""
    for lang in load_languages():
        if lang.is_code_mixed:
            assert lang.script == "Latin"
            assert lang.base_language


def test_resource_ranks_are_unique_and_ordered():
    langset = load_languages()
    ranks = [lang.resource_rank for lang in langset]
    assert len(set(ranks)) == len(ranks)
    assert langset.by_code("en").resource_rank == 0
    assert langset.by_code("hi").resource_rank < langset.by_code("or").resource_rank


# -- model configs ----------------------------------------------------------------------


def test_sarvam_config_matches_verified_architecture():
    """Guards the numbers the whole design rests on. If a Hub revision changes these, the
    hook design and the parameter budget both need revisiting."""
    arch = load_model_config("sarvam-30b")["architecture"]
    assert arch["num_hidden_layers"] == 19
    assert arch["first_k_dense_replace"] == 1
    assert arch["num_experts"] == 128
    assert arch["num_experts_per_tok"] == 6
    assert arch["num_shared_experts"] == 1
    assert arch["hidden_size"] == 4096
    assert arch["vocab_size"] == 262144


def test_tiny_config_mirrors_the_real_topology():
    """The tiny model must have a dense layer 0 and MoE layers above it, like the real one —
    otherwise it does not exercise the layer-index logic the hooks depend on."""
    tiny = load_model_config("sarvam-30b-tiny")["architecture"]
    assert tiny["first_k_dense_replace"] == 1
    assert tiny["num_hidden_layers"] > 1
    assert tiny["num_experts_per_tok"] < tiny["num_experts"]


def test_expert_share_dominates_the_parameter_budget():
    """~90% of sarvam-30b is routed experts. This is why quantizing it IS quantizing the
    experts, and why the coverage mechanism matters."""
    from indicquant.quant.sarvam_moe_adapter import expert_parameter_share

    share = expert_parameter_share(load_model_config("sarvam-30b"))
    assert share["routed_experts"] > 0.85
    assert share["router_gate"] < 0.01
    total = load_model_config("sarvam-30b")["architecture"]["total_parameters"]
    assert abs(share["total_parameters"] - total) / total < 0.02


def test_tokens_per_expert_fraction():
    from indicquant.quant.sarvam_moe_adapter import tokens_per_expert_fraction

    assert tokens_per_expert_fraction(load_model_config("sarvam-30b")) == pytest.approx(6 / 128)
