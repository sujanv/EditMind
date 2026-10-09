"""Abstract base class for all knowledge editors."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import copy
import time
import uuid

from editmind.core.types import EditRequest, EditResult


class BaseKnowledgeEditor(ABC):
    """Abstract Base Class for LLM Knowledge Editing methods."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        self.model_wrapper = model_wrapper
        self.config = config or {}
        self.edit_history: List[EditResult] = []
        self._checkpoints: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the identifier name of the editor."""
        pass

    @abstractmethod
    def edit(self, request: EditRequest) -> EditResult:
        """Performs a single knowledge edit on the model.
        
        Args:
            request: The EditRequest containing target prompt and new assertion.

        Returns:
            EditResult containing execution metrics and outcome.
        """
        pass

    def batch_edit(self, requests: List[EditRequest]) -> List[EditResult]:
        """Performs batch or sequential editing across multiple requests.
        
        Subclasses may override this for vectorized or batched optimizations.
        """
        results = []
        for req in requests:
            res = self.edit(req)
            results.append(res)
        return results

    def create_checkpoint(self) -> str:
        """Saves current model state to an internal checkpoint."""
        ckpt_id = str(uuid.uuid4())[:8]
        state = self.model_wrapper.get_state_dict()
        # Deepcopy state tensors
        self._checkpoints[ckpt_id] = {k: v.clone() if hasattr(v, "clone") else copy.deepcopy(v) 
                                      for k, v in state.items()}
        return ckpt_id

    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        """Restores model weights to the specified checkpoint (or most recent)."""
        if not self._checkpoints:
            return False
        if checkpoint_id is None:
            checkpoint_id = list(self._checkpoints.keys())[-1]
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint ID '{checkpoint_id}' not found.")
        self.model_wrapper.load_state_dict(self._checkpoints[checkpoint_id])
        return True

    def get_edit_history(self) -> List[EditResult]:
        """Returns the sequential history of performed edits."""
        return list(self.edit_history)

    def measure_target_probabilities(self, prompt: str, target: str, old_target: Optional[str] = None):
        """Measures probabilities of target and old_target given prompt."""
        probs = self.model_wrapper.predict_next_token_probs(prompt)
        p_target = probs.get(target, 0.0)
        p_old = probs.get(old_target, 0.0) if old_target else 0.0
        return p_target, p_old
