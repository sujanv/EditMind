"""Machine Unlearning for LLMs (Privacy erasure and safety alignment)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import torch
import torch.nn.functional as F

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry


@dataclass
class UnlearnResult:
    """Contains diagnostics from machine unlearning execution."""
    prompt: str
    target_erased: str
    success: bool
    pre_prob: float
    post_prob: float
    probability_reduction: float
    execution_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target_erased": self.target_erased,
            "success": self.success,
            "pre_prob": round(self.pre_prob, 5),
            "post_prob": round(self.post_prob, 5),
            "probability_reduction": round(self.probability_reduction, 4),
            "execution_time_sec": round(self.execution_time_sec, 4),
        }


class MachineUnlearner:
    """Selectively unlearns private facts, PII, and unsafe memories from LLM weights."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, editor_name: str = "rome"):
        self.wrapper = model_wrapper
        self.editor = EditorRegistry.create(editor_name, model_wrapper=self.wrapper)

    def unlearn_fact(
        self,
        prompt: str,
        target_to_erase: str,
        subject: Optional[str] = None,
        replacement_token: str = "unknown",
    ) -> UnlearnResult:
        """Erases memory of target_to_erase by redirecting activation flow to a safe neutral token."""
        start_time = time.time()

        # Measure baseline probability of target to erase
        probs_pre = self.wrapper.predict_next_token_probs(prompt)
        p_pre = probs_pre.get(target_to_erase, 0.0)

        # Create edit request steering target to safe replacement
        req = EditRequest(
            prompt=prompt,
            target_new=replacement_token,
            ground_truth=target_to_erase,
            subject=subject,
            metadata={"task": "machine_unlearning"}
        )

        res = self.editor.edit(req)

        probs_post = self.wrapper.predict_next_token_probs(prompt)
        p_post = probs_post.get(target_to_erase, 0.0)

        reduction = max(0.0, p_pre - p_post) / max(p_pre, 1e-5)
        elapsed = time.time() - start_time
        success = p_post < p_pre or p_post < 0.01

        return UnlearnResult(
            prompt=prompt,
            target_erased=target_to_erase,
            success=success,
            pre_prob=p_pre,
            post_prob=p_post,
            probability_reduction=reduction,
            execution_time_sec=elapsed,
        )
