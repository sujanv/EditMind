"""Knowledge conflict detection and consistency validation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
import networkx as nx

from editmind.core.types import EditRequest, KnowledgeTriple


@dataclass
class ConflictReport:
    """Contains diagnostics on detected knowledge conflicts."""
    has_conflict: bool
    conflict_type: Optional[str] = None  # "contradiction", "circular_dependency", "redundant"
    conflicting_requests: List[str] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type,
            "conflicting_requests": self.conflicting_requests or [],
            "explanation": self.explanation,
        }


class KnowledgeConflictDetector:
    """Detects semantic contradictions and cyclical inconsistencies among knowledge edits."""

    def __init__(self):
        self.active_facts: Dict[str, KnowledgeTriple] = {}  # key: (subject, relation)
        self.graph = nx.DiGraph()

    def check_request(self, request: EditRequest) -> ConflictReport:
        """Evaluates whether an incoming EditRequest contradicts or conflicts with existing facts."""
        subj = (request.subject or "").strip().lower()
        rel = (request.relation or "").strip().lower()
        target = request.target_new.strip().lower()

        if not subj or not rel:
            return ConflictReport(has_conflict=False, explanation="No formal subject/relation defined.")

        fact_key = f"{subj}::{rel}"

        # 1. Contradiction check: Same (subject, relation), conflicting target
        if fact_key in self.active_facts:
            existing = self.active_facts[fact_key]
            if existing.target.lower() != target:
                return ConflictReport(
                    has_conflict=True,
                    conflict_type="contradiction",
                    conflicting_requests=[str(existing), f"({request.subject}, {request.relation}, {request.target_new})"],
                    explanation=f"Direct contradiction: '{request.subject} {request.relation}' already mapped to '{existing.target}'."
                )

        # 2. Cycle detection in relation graph (e.g. parent_of / located_in loops)
        if rel in ["located in", "part of", "subsidiary of", "capital of"]:
            self.graph.add_edge(subj, target)
            if not nx.is_directed_acyclic_graph(self.graph):
                self.graph.remove_edge(subj, target)
                return ConflictReport(
                    has_conflict=True,
                    conflict_type="circular_dependency",
                    conflicting_requests=[f"{subj} -> {target}"],
                    explanation=f"Cycle detected: adding relation '{subj} {rel} {target}' creates circular dependency."
                )
            self.graph.remove_edge(subj, target)

        return ConflictReport(has_conflict=False, explanation="No conflict detected.")

    def register_fact(self, request: EditRequest):
        """Adds verified fact to active knowledge registry."""
        subj = (request.subject or "").strip().lower()
        rel = (request.relation or "").strip().lower()
        if subj and rel:
            fact_key = f"{subj}::{rel}"
            triple = KnowledgeTriple(
                subject=request.subject,
                relation=request.relation,
                target=request.target_new,
                old_target=request.ground_truth,
            )
            self.active_facts[fact_key] = triple
            if rel in ["located in", "part of", "subsidiary of", "capital of"]:
                self.graph.add_edge(subj, request.target_new.strip().lower())
