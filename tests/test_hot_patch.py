"""Unit tests for Dynamic Knowledge Hot-Patching Engine."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.hot_patch import HotPatch, HotPatchEngine


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_hot_patch_lifecycle(toy_wrapper):
    engine = HotPatchEngine(toy_wrapper)
    target_layer = 1
    w_orig = toy_wrapper.model.layers[target_layer].mlp.down_proj.weight.clone()

    # Create synthetic delta tensor [hidden_dim, intermediate_dim]
    shape = w_orig.shape
    synthetic_delta = torch.randn(shape) * 0.1

    # 1. Register and apply patch
    patch = engine.register_patch(
        subject="Eiffel Tower",
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        layer_idx=target_layer,
        delta_tensor=synthetic_delta,
        scaling=1.0,
        patch_id="p1",
    )

    assert isinstance(patch, HotPatch)
    assert patch.enabled is True
    assert patch.patch_id == "p1"

    # Weight should now differ from original
    w_patched = toy_wrapper.model.layers[target_layer].mlp.down_proj.weight
    assert not torch.allclose(w_patched, w_orig)

    # 2. Hot-toggle disable
    changed = engine.hot_toggle("p1", enabled=False)
    assert changed is True
    assert patch.enabled is False
    # Weight should be restored back to original!
    assert torch.allclose(w_patched, w_orig, atol=1e-5)

    # 3. Hot-toggle re-enable
    changed = engine.hot_toggle("p1", enabled=True)
    assert changed is True
    assert patch.enabled is True
    assert not torch.allclose(w_patched, w_orig)

    # 4. Scaling adjustment
    engine.set_patch_scaling("p1", scaling=2.0)
    assert patch.scaling == 2.0

    # 5. Delete patch
    deleted = engine.delete_patch("p1")
    assert deleted is True
    assert "p1" not in engine.patches
    assert torch.allclose(w_patched, w_orig, atol=1e-5)


def test_multi_patch_interference(toy_wrapper):
    engine = HotPatchEngine(toy_wrapper)
    target_layer = 2
    shape = toy_wrapper.model.layers[target_layer].mlp.down_proj.weight.shape

    d1 = torch.randn(shape) * 0.05
    d2 = torch.randn(shape) * 0.05

    engine.register_patch("Paris", "prompt1", "Rome", target_layer, d1, patch_id="pA")
    engine.register_patch("London", "prompt2", "Berlin", target_layer, d2, patch_id="pB")

    summary = engine.get_summary()
    assert summary["total_patches"] == 2
    assert summary["active_patches"] == 2

    matrix_info = engine.compute_patch_interference_matrix()
    assert len(matrix_info["patch_ids"]) == 2
    assert matrix_info["matrix"][0][0] == 1.0
    assert matrix_info["matrix"][1][1] == 1.0
    assert -1.0 <= matrix_info["matrix"][0][1] <= 1.0
