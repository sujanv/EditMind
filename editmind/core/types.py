"""Core types and data representations for EditMind."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time
import uuid


@dataclass
class KnowledgeTriple:
    """Represents a factual knowledge relation (Subject, Relation, Object)."""
    subject: str
    relation: str
    target: str
    old_target: Optional[str] = None

    def __str__(self) -> str:
        return f"({self.subject}, {self.relation}, {self.target})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeTriple:
        return cls(**data)


@dataclass
class EditRequest:
    """Encapsulates a single knowledge editing request with verification probes."""
    prompt: str
    target_new: str
    ground_truth: Optional[str] = None
    subject: Optional[str] = None
    relation: Optional[str] = None
    rephrase_prompts: List[str] = field(default_factory=list)
    locality_prompts: List[Dict[str, str]] = field(default_factory=list)
    portability_prompts: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EditRequest:
        return cls(**data)

    @property
    def triple(self) -> Optional[KnowledgeTriple]:
        if self.subject and self.relation and self.target_new:
            return KnowledgeTriple(
                subject=self.subject,
                relation=self.relation,
                target=self.target_new,
                old_target=self.ground_truth,
            )
        return None


@dataclass
class EditResult:
    """Contains outcome, diagnostic metrics, and verification scores of an edit."""
    request_id: str
    editor_name: str
    success: bool
    execution_time_sec: float
    pre_edit_target_prob: float = 0.0
    post_edit_target_prob: float = 0.0
    pre_edit_old_prob: float = 0.0
    post_edit_old_prob: float = 0.0
    delta_weight_norm: float = 0.0
    checkpoint_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EditResult:
        return cls(**data)


@dataclass
class EvaluationMetrics:
    """Multi-dimensional evaluation metrics for knowledge editing quality."""
    efficacy: float = 0.0            # P(target_new) > P(target_old) on edit prompt
    generality: float = 0.0          # Success rate across paraphrase prompts
    locality: float = 0.0            # Retention rate on unrelated neighborhood facts
    portability: float = 0.0         # Accuracy on downstream multi-hop/compositional prompts
    ripple_retention: float = 0.0    # Memory retention across sequential edits
    edit_latency_ms: float = 0.0     # Wall-clock editing time in ms
    delta_norm: float = 0.0          # L2 norm of parameter change

    @property
    def composite_score(self) -> float:
        """Harmonic balance across Efficacy, Generality, and Locality."""
        components = [self.efficacy, self.generality, self.locality]
        if any(c <= 0.0 for c in components):
            return 0.0
        return 3.0 / (1.0 / self.efficacy + 1.0 / self.generality + 1.0 / self.locality)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["composite_score"] = round(self.composite_score, 4)
        return d


@dataclass
class ModelOutput:
    """Standardized causal model output container."""
    logits: Any
    loss: Optional[float] = None
    hidden_states: Optional[List[Any]] = None
    attentions: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    token_ids: Optional[List[int]] = None
