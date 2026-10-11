import os
import tempfile
import pytest
import torch

from editmind.core.types import KnowledgeTriple
from editmind.core.checkpoint import CheckpointManager, EditPatch
from editmind.models.model_wrapper import UnifiedModelWrapper


def test_checkpoint_capture_and_rollback():
    model_orig = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    model_edited = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)

    # Simulate an edit modifying a weight tensor
    with torch.no_grad():
        target_param = next(model_edited.model.parameters())
        target_param.add_(0.5)

    triple = KnowledgeTriple(subject="Paris", relation="capital of", target="France")
    mgr = CheckpointManager()

    patch = mgr.capture_deltas(
        model_before=model_orig,
        model_after=model_edited,
        editor_name="ROME",
        target_triple=triple,
        patch_id="patch_test_1",
    )

    assert patch.patch_id == "patch_test_1"
    assert patch.verify_integrity() is True
    assert len(patch.layer_deltas) > 0

    # Verify applying rollback restores the exact original parameters
    mgr.rollback_patch(model_edited, patch)
    p_restored = next(model_edited.model.parameters())
    p_orig = next(model_orig.model.parameters())
    assert torch.allclose(p_restored, p_orig, atol=1e-6)

    # Re-apply forward
    mgr.apply_patch(model_edited, patch)
    assert not torch.allclose(next(model_edited.model.parameters()), p_orig, atol=1e-4)


def test_checkpoint_serialization_roundtrip():
    model_orig = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    model_edited = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)

    with torch.no_grad():
        target_param = next(model_edited.model.parameters())
        target_param.add_(0.25)

    triple = KnowledgeTriple(subject="London", relation="capital of", target="UK")
    mgr = CheckpointManager()
    patch = mgr.capture_deltas(model_orig, model_edited, "MEMIT", triple)

    with tempfile.TemporaryDirectory() as tmpdir:
        patch_path = os.path.join(tmpdir, "test.editpatch")
        mgr.save_patch(patch, patch_path)
        assert os.path.exists(patch_path)

        loaded_patch = mgr.load_patch(patch_path)
        assert loaded_patch.patch_id == patch.patch_id
        assert loaded_patch.sha256 == patch.sha256
        assert loaded_patch.verify_integrity() is True
