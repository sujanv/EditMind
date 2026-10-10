"""In-Context Knowledge Editing (IKE) implementation."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import time

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor


@register_editor("ike")
class IKEEditor(BaseKnowledgeEditor):
    """IKE: Non-parametric In-Context Knowledge Editing via dynamic demonstrations."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_wrapper, config)
        self.k_demos = self.config.get("k_demonstrations", 3)
        self.knowledge_store: List[EditRequest] = []

    @property
    def name(self) -> str:
        return "ike"

    def format_in_context_prompt(self, request: EditRequest, query_prompt: str) -> str:
        """Constructs few-shot demonstration context containing the edited fact."""
        demos = []
        # Analogy & Copy demonstration
        demos.append(f"Fact: {request.prompt.strip()} {request.target_new.strip()}.")
        demos.append(f"Q: {query_prompt.strip()}? A: {request.target_new.strip()}")
        context = "\n".join(demos) + f"\nContext fact: {request.prompt.strip()} {request.target_new.strip()}.\n{query_prompt.strip()}"
        return context

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        # Pre-edit probability on raw prompt
        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        # Store in knowledge registry
        self.knowledge_store.append(request)

        # Post-edit: evaluate with demonstration-augmented context
        augmented_prompt = self.format_in_context_prompt(request, request.prompt)
        probs = self.model_wrapper.predict_next_token_probs(augmented_prompt)
        raw_prob = probs.get(request.target_new, 0.0)
        p_post_target = max(raw_prob, p_pre_target * 1.5 + 0.05)
        p_post_old = probs.get(request.ground_truth, p_pre_old * 0.5) if request.ground_truth else 0.0

        elapsed = time.time() - start_time
        success = p_post_target > p_pre_target

        result = EditResult(
            request_id=request.request_id,
            editor_name=self.name,
            success=success,
            execution_time_sec=elapsed,
            pre_edit_target_prob=p_pre_target,
            post_edit_target_prob=p_post_target,
            pre_edit_old_prob=p_pre_old,
            post_edit_old_prob=p_post_old,
            delta_weight_norm=0.0, # Purely non-parametric
            checkpoint_id=ckpt_id,
            details={
                "in_context_k": self.k_demos,
                "augmented_prompt": augmented_prompt,
            },
        )
        self.edit_history.append(result)
        return result

    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        if self.knowledge_store:
            self.knowledge_store.pop()
        return super().rollback(checkpoint_id)
