import pytest
import torch
from editmind.core.registry import EditorRegistry
from editmind.core.types import EditRequest
from editmind.editors.qedit import (
    QuantizedKnowledgeEditor,
    QuantizedLinearAdapter,
    simulate_weight_quantization,
)
from editmind.models.model_wrapper import UnifiedModelWrapper


def test_simulate_weight_quantization():
    tensor = torch.randn(10, 10)
    q4 = simulate_weight_quantization(tensor, bits=4)
    assert q4.shape == tensor.shape
    # Unique values should be bounded by 2^4 = 16
    assert len(torch.unique(q4)) <= 16

    q8 = simulate_weight_quantization(tensor, bits=8)
    assert q8.shape == tensor.shape
    assert len(torch.unique(q8)) <= 256


def test_quantized_linear_adapter_forward():
    adapter = QuantizedLinearAdapter(in_features=32, out_features=64, rank=4)
    x = torch.randn(2, 5, 32)
    out = adapter(x)
    assert out.shape == (2, 5, 64)


def test_qedit_knowledge_editing():
    model = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    editor = EditorRegistry.create("qedit", model_wrapper=model, rank=4, bits=4, steps=20, lr=0.1)

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    result = editor.edit(req)
    assert result.success is True
    assert result.editor_name == "Q-Edit"
    assert result.post_edit_target_prob > result.pre_edit_target_prob
    assert result.delta_weight_norm > 0.0
    assert result.details["quantized_bits"] == 4
