"""Unit tests for UnifiedModelWrapper, ToyCausalLM, hooks, and CausalTracer."""

import pytest
import torch
from editmind.models import (
    UnifiedModelWrapper,
    ToyCausalLM,
    SimpleTokenizer,
    CausalTracer,
    ActivationCacher,
    hook_activation_intervention,
)


def test_tokenizer():
    tok = SimpleTokenizer()
    encoded = tok.encode("The Eiffel Tower is in Paris")
    assert len(encoded) == 6
    decoded = tok.decode(encoded)
    assert "Eiffel" in decoded


def test_toy_causal_lm_forward():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    prompt = "The Eiffel Tower is in"
    probs = wrapper.predict_next_token_probs(prompt)
    assert isinstance(probs, dict)
    assert "Paris" in probs
    assert 0.0 <= probs["Paris"] <= 1.0


def test_activation_cacher():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    modules = {"layer_0": wrapper.model.layers[0], "layer_1": wrapper.model.layers[1]}
    with ActivationCacher(modules) as cacher:
        input_ids = wrapper.tokenize("The Eiffel Tower")
        wrapper.forward(input_ids)

    assert "layer_0" in cacher.activations
    assert len(cacher.activations["layer_0"]) == 1
    assert cacher.activations["layer_0"][0].shape[-1] == 64


def test_activation_intervention():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    target_mod = wrapper.model.layers[2]
    
    flag = {"called": False}
    def intervention_fn(x):
        flag["called"] = True
        return x * 0.5

    with hook_activation_intervention(target_mod, intervention_fn):
        input_ids = wrapper.tokenize("Testing hooks")
        wrapper.forward(input_ids)

    assert flag["called"] is True


def test_causal_tracer():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    tracer = CausalTracer(wrapper, noise_std=0.2)
    result = tracer.trace(
        prompt="The Eiffel Tower is in",
        subject="Eiffel Tower",
        target="Paris",
    )
    assert result.indirect_effects.shape == (4, len(result.tokens))
    best_layer, best_token, token_str = result.find_critical_layer_and_token()
    assert 0 <= best_layer < 4
    assert 0 <= best_token < len(result.tokens)
    dict_repr = result.to_dict()
    assert "indirect_effects" in dict_repr
