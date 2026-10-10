"""Cross-lingual knowledge transfer evaluator."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest
from editmind.multilingual.dataset import MultilingualFact


@dataclass
class CrossLingualTransferReport:
    """Diagnostic report on knowledge propagation across languages."""
    fact_subject: str
    target: str
    transfer_scores: Dict[str, float]  # lang -> success (0.0 or 1.0)
    average_transfer_rate: float
    disparity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_subject": self.fact_subject,
            "target": self.target,
            "transfer_scores": self.transfer_scores,
            "average_transfer_rate": round(self.average_transfer_rate, 4),
            "disparity_score": round(self.disparity_score, 4),
        }


class CrossLingualEvaluator:
    """Measures whether edits made in one language propagate across parallel multilingual queries."""

    def __init__(self, editor: BaseKnowledgeEditor):
        self.editor = editor
        self.wrapper = editor.model_wrapper

    def evaluate_fact_transfer(self, fact: MultilingualFact) -> CrossLingualTransferReport:
        """Applies the edit using English prompt, then probes in parallel languages."""
        req = EditRequest(
            prompt=fact.english_prompt,
            target_new=fact.target_new,
            ground_truth=fact.ground_truth,
            subject=fact.subject,
        )

        # Apply edit
        self.editor.edit(req)

        scores: Dict[str, float] = {}
        for lang, prompt in fact.parallel_prompts.items():
            probs = self.wrapper.predict_next_token_probs(prompt)
            p_new = probs.get(fact.target_new, 0.0)
            p_old = probs.get(fact.ground_truth, 0.0)
            scores[lang] = 1.0 if p_new >= p_old else 0.0

        rates = list(scores.values())
        avg_transfer = float(np.mean(rates)) if rates else 0.0
        disparity = float(np.std(rates)) if rates else 0.0

        return CrossLingualTransferReport(
            fact_subject=fact.subject,
            target=fact.target_new,
            transfer_scores=scores,
            average_transfer_rate=avg_transfer,
            disparity_score=disparity,
        )
