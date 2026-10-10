"""Unit tests for ContinualKnowledgeEditor and Interference Matrix."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.registry import EditorRegistry
from editmind.data import load_counterfact_dataset
from editmind.continual import (
    ContinualKnowledgeEditor,
    render_ascii_interference_matrix,
    generate_svg_interference_matrix,
)


def test_continual_lifelong_editing():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = EditorRegistry.create("grace", model_wrapper=wrapper, config={"target_layer": 2, "inner_steps": 10})
    continual = ContinualKnowledgeEditor(editor)

    dataset = load_counterfact_dataset(sample_count=3)
    trajectory = continual.run_sequential_stream(dataset.requests)

    assert trajectory.interference_matrix.shape == (3, 3)
    assert 0.0 <= trajectory.average_retention <= 1.0
    assert trajectory.catastrophic_forgetting_rate >= 0.0

    ascii_rep = render_ascii_interference_matrix(trajectory)
    assert "CONTINUAL EDIT INTERFERENCE MATRIX" in ascii_rep

    svg_rep = generate_svg_interference_matrix(trajectory)
    assert "<svg" in svg_rep
    assert "Step 0" in svg_rep
