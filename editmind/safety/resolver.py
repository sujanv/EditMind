"""Automated Knowledge Conflict Resolution and Temporal Scoping Engine.

Provides policy-based resolution for contradictory edits, cyclic dependencies,
and temporal fact transitions (e.g., latest-wins, confidence-weighting,
multi-valued attribute merging, and temporal validity intervals).
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Set

from editmind.core.types import EditRequest, KnowledgeTriple
from editmind.safety.conflict_detector import KnowledgeConflictDetector, ConflictReport


class ResolutionStrategy(str, Enum):
    LATEST_WINS = "latest_wins"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    TEMPORAL_SCOPING = "temporal_scoping"
    MERGE_MULTI_VALUED = "merge_multi_valued"
    REJECT_ON_CONFLICT = "reject_on_conflict"


@dataclass
class ScopedKnowledgeFact:
    """Represents a fact enriched with temporal scope and versioning."""
    subject: str
    relation: str
    target: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    confidence: float = 1.0
    version: int = 1
    superseded_by: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ResolutionDecision:
    """Outcome of automated conflict resolution."""
    action: str  # "ACCEPTED", "SUPERSEDED", "MERGED", "REJECTED", "TEMPORAL_SPLIT"
    strategy_applied: ResolutionStrategy
    effective_target: str
    reason: str
    resolved_fact: ScopedKnowledgeFact
    audit_notes: List[str] = field(default_factory=list)


# Relations that naturally support multiple concurrent targets
MULTI_VALUED_RELATIONS = {
    "spoken language",
    "languages spoken",
    "author of",
    "starred in",
    "member of",
    "borders",
    "synonym of",
    "genre",
}


class KnowledgeConflictResolver:
    """Automates resolution of factual contradictions and dependency cycles."""

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.LATEST_WINS,
        conflict_detector: Optional[KnowledgeConflictDetector] = None,
    ):
        self.default_strategy = default_strategy
        self.detector = conflict_detector or KnowledgeConflictDetector()
        self.knowledge_base: Dict[str, ScopedKnowledgeFact] = {}  # key: "subject::relation"

    def _make_key(self, subject: str, relation: str) -> str:
        return f"{subject.lower().strip()}::{relation.lower().strip()}"

    def resolve(
        self,
        request: EditRequest,
        strategy: Optional[ResolutionStrategy] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
    ) -> ResolutionDecision:
        """Resolves conflict between incoming edit request and existing knowledge base."""
        strat = strategy or self.default_strategy
        subj = (request.subject or "").strip()
        rel = (request.relation or "associated with").strip()
        target = request.target_new.strip()
        key = self._make_key(subj, rel)

        existing = self.knowledge_base.get(key)

        # 1. No conflict: first time seeing this subject-relation
        if existing is None:
            new_fact = ScopedKnowledgeFact(
                subject=subj,
                relation=rel,
                target=target,
                valid_from=valid_from,
                valid_to=valid_to,
                confidence=confidence,
                version=1,
            )
            self.knowledge_base[key] = new_fact
            self.detector.register_fact(request)
            return ResolutionDecision(
                action="ACCEPTED",
                strategy_applied=strat,
                effective_target=target,
                reason="No prior fact exists for subject-relation pair.",
                resolved_fact=new_fact,
            )

        # 2. Existing fact matches incoming target (idempotent update)
        if existing.target.lower() == target.lower():
            return ResolutionDecision(
                action="ACCEPTED",
                strategy_applied=strat,
                effective_target=target,
                reason="Incoming fact is identical to active fact.",
                resolved_fact=existing,
            )

        # 3. Detect multi-valued relation merging
        if strat == ResolutionStrategy.MERGE_MULTI_VALUED or rel.lower() in MULTI_VALUED_RELATIONS:
            existing_targets = {t.strip() for t in existing.target.split(",")}
            existing_targets.add(target)
            merged_target = ", ".join(sorted(existing_targets))
            existing.target = merged_target
            existing.version += 1
            return ResolutionDecision(
                action="MERGED",
                strategy_applied=ResolutionStrategy.MERGE_MULTI_VALUED,
                effective_target=merged_target,
                reason=f"Multi-valued relation '{rel}' merged with existing targets.",
                resolved_fact=existing,
                audit_notes=[f"Previous targets: {existing.target}", f"Added target: {target}"],
            )

        # 4. Temporal scoping
        if strat == ResolutionStrategy.TEMPORAL_SCOPING or valid_from is not None:
            existing.superseded_by = target
            existing.valid_to = valid_from or str(time.strftime("%Y"))
            scoped_fact = ScopedKnowledgeFact(
                subject=subj,
                relation=rel,
                target=target,
                valid_from=valid_from,
                valid_to=valid_to,
                confidence=confidence,
                version=existing.version + 1,
            )
            self.knowledge_base[key] = scoped_fact
            return ResolutionDecision(
                action="TEMPORAL_SPLIT",
                strategy_applied=ResolutionStrategy.TEMPORAL_SCOPING,
                effective_target=target,
                reason=f"Temporal partition: previous valid until {existing.valid_to}; new valid from {valid_from}.",
                resolved_fact=scoped_fact,
                audit_notes=[f"Prior fact: {existing.target} (<= {existing.valid_to})"],
            )

        # 5. Confidence-weighted resolution
        if strat == ResolutionStrategy.CONFIDENCE_WEIGHTED:
            if confidence >= existing.confidence:
                existing.superseded_by = target
                new_fact = ScopedKnowledgeFact(
                    subject=subj,
                    relation=rel,
                    target=target,
                    confidence=confidence,
                    version=existing.version + 1,
                )
                self.knowledge_base[key] = new_fact
                return ResolutionDecision(
                    action="SUPERSEDED",
                    strategy_applied=strat,
                    effective_target=target,
                    reason=f"Incoming confidence ({confidence:.2f}) >= existing confidence ({existing.confidence:.2f}).",
                    resolved_fact=new_fact,
                )
            else:
                return ResolutionDecision(
                    action="REJECTED",
                    strategy_applied=strat,
                    effective_target=existing.target,
                    reason=f"Incoming confidence ({confidence:.2f}) < existing confidence ({existing.confidence:.2f}).",
                    resolved_fact=existing,
                )

        # 6. Reject on conflict strategy
        if strat == ResolutionStrategy.REJECT_ON_CONFLICT:
            return ResolutionDecision(
                action="REJECTED",
                strategy_applied=strat,
                effective_target=existing.target,
                reason=f"Conflict detected with active fact: '{existing.subject} {existing.relation} {existing.target}'.",
                resolved_fact=existing,
            )

        # 7. Default: Latest Wins (Temporal Override)
        prior_target = existing.target
        existing.superseded_by = target
        new_fact = ScopedKnowledgeFact(
            subject=subj,
            relation=rel,
            target=target,
            confidence=confidence,
            version=existing.version + 1,
        )
        self.knowledge_base[key] = new_fact
        self.detector.register_fact(request)
        return ResolutionDecision(
            action="SUPERSEDED",
            strategy_applied=ResolutionStrategy.LATEST_WINS,
            effective_target=target,
            reason=f"Superseded prior target '{prior_target}' via latest-wins policy.",
            resolved_fact=new_fact,
            audit_notes=[f"Previous target: {prior_target}"],
        )
