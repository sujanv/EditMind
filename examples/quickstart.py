"""Minimal 10-line Quickstart for EditMind."""

from editmind.models import UnifiedModelWrapper
from editmind.core.registry import EditorRegistry
from editmind.core.types import EditRequest

# 1. Load model wrapper and editor
wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
editor = EditorRegistry.create("rome", model_wrapper=wrapper)

# 2. Define fact to edit
request = EditRequest(
    prompt="The Eiffel Tower is in",
    target_new="Rome",
    ground_truth="Paris",
    subject="Eiffel Tower",
)

# 3. Apply edit
result = editor.edit(request)
print(f"Edit {result.editor_name}: Success={result.success}, Post-P={result.post_edit_target_prob:.4f}")
