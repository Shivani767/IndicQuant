"""Router hook tests against the real SarvamMoE code, on CPU.

These run on the tiny random-init model, which instantiates the actual `SarvamMoEGate` class
from sarvam's remote code. That means these tests would catch a change to the gate's return
signature — the thing the entire RQ4 design rests on — before it costs GPU time.
"""

from __future__ import annotations

import pytest

from indicquant.config import load_model_config
from indicquant.models import count_moe_layers
from indicquant.routing.hooks import (
    RoutingCaptureError,
    _layer_index_from_name,
    find_gate_modules,
    sequence_fingerprint,
)

# -- pure helpers, no model needed ------------------------------------------------------


def test_layer_index_parsing():
    assert _layer_index_from_name("model.layers.7.mlp.gate") == 7
    assert _layer_index_from_name("model.layers.18.mlp.gate") == 18
    assert _layer_index_from_name("lm_head") is None
    assert _layer_index_from_name("model.embed_tokens") is None


def test_moe_layer_indices_skip_the_dense_layer():
    """sarvam-30b has first_k_dense_replace=1, so MoE layers are 1..18, not 0..18."""
    cfg = load_model_config("sarvam-30b")
    layers = count_moe_layers(cfg)
    assert layers == list(range(1, 19))
    assert len(layers) == 18
    assert 0 not in layers


def test_fingerprint_is_stable_and_discriminating():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[1, 2, 4]])
    assert sequence_fingerprint(a) == sequence_fingerprint(a.clone())
    assert sequence_fingerprint(a) != sequence_fingerprint(b)


def test_find_gate_modules_on_a_dense_model_returns_nothing():
    torch = pytest.importorskip("torch")

    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])

    assert find_gate_modules(Dense()) == {}


# -- against the real Sarvam gate -------------------------------------------------------


@pytest.mark.network
def test_hooks_bind_to_every_moe_layer(tiny_model):
    cfg = load_model_config("sarvam-30b-tiny")
    gates = find_gate_modules(tiny_model)
    assert sorted(gates) == count_moe_layers(cfg)


@pytest.mark.network
def test_recorder_captures_routing_end_to_end(tiny_model, tmp_path):
    torch = pytest.importorskip("torch")
    from indicquant.routing.hooks import RouterRecorder
    from indicquant.routing.storage import read_trace

    cfg = load_model_config("sarvam-30b-tiny")
    arch = cfg["architecture"]
    batch, seq_len = 2, 16
    input_ids = torch.randint(0, arch["vocab_size"] - 1, (batch, seq_len))

    with RouterRecorder(tiny_model, out_dir=tmp_path, condition="A", language="hi") as rec:
        rec.set_sequence(input_ids, seq_ids=[0, 1])
        with torch.no_grad():
            tiny_model(input_ids=input_ids)

    trace = read_trace(tmp_path, condition="A", language="hi")
    n_moe_layers = len(count_moe_layers(cfg))
    assert len(trace) == batch * seq_len * n_moe_layers
    assert set(trace["layer"]) == set(count_moe_layers(cfg))
    assert trace["expert_ids"].apply(len).eq(arch["num_experts_per_tok"]).all()
    assert trace["expert_ids"].apply(lambda e: max(e) < arch["num_experts"]).all()
    assert trace["fingerprint"].nunique() == 1


@pytest.mark.network
def test_recorder_requires_set_sequence(tiny_model, tmp_path):
    """Forgetting set_sequence must fail loudly — a trace without fingerprints cannot be
    compared across conditions, and discovering that after a GPU run is expensive."""
    torch = pytest.importorskip("torch")
    from indicquant.routing.hooks import RouterRecorder

    input_ids = torch.randint(0, 999, (1, 8))
    with RouterRecorder(tiny_model, out_dir=tmp_path, condition="A", language="hi"):
        with pytest.raises(RoutingCaptureError, match="set_sequence"), torch.no_grad():
            tiny_model(input_ids=input_ids)


@pytest.mark.network
def test_hooks_are_removed_on_exit(tiny_model, tmp_path):
    from indicquant.routing.hooks import RouterRecorder

    gate = next(iter(find_gate_modules(tiny_model).values()))
    before = len(gate._forward_hooks)
    with RouterRecorder(tiny_model, out_dir=tmp_path, condition="A", language="hi"):
        assert len(gate._forward_hooks) == before + 1
    assert len(gate._forward_hooks) == before


def test_recorder_raises_when_no_gates_found(tmp_path):
    """A fused backend (vLLM, llama.cpp) binds zero hooks. That must raise rather than
    produce an empty trace that reads as 'no routing drift'."""
    torch = pytest.importorskip("torch")
    from indicquant.routing.hooks import RouterRecorder

    model = torch.nn.Linear(4, 4)
    with pytest.raises(RoutingCaptureError, match="no gate modules"):
        with RouterRecorder(model, out_dir=tmp_path, condition="A", language="hi"):
            pass
