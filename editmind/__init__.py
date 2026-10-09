"""
EditMind: Unified Knowledge Editing Framework for Large Language Models.

EditMind enables precise, localized modification of factual memory in LLMs
without expensive retraining or catastrophic forgetting across different editing paradigms:
- Locate-and-Edit (ROME, MEMIT)
- Meta-learning / Hypernetworks (MEND)
- Explicit Memory / Activation Cache (GRACE)
- Non-parametric In-Context Retrieval (IKE)
- Constrained Fine-Tuning Baselines (FT-L, LoRA-Edit)
"""

__version__ = "0.1.0"
__author__ = "Sujan Venkat"

from editmind.core.types import (
    EditRequest,
    EditResult,
    KnowledgeTriple,
    EvaluationMetrics,
)
from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.registry import EditorRegistry, register_editor

__all__ = [
    "EditRequest",
    "EditResult",
    "KnowledgeTriple",
    "EvaluationMetrics",
    "BaseKnowledgeEditor",
    "EditorRegistry",
    "register_editor",
]
