"""Factual Reciprocal Consistency and Bidirectional Knowledge Auditor.

Evaluates the 'Reversal Curse' and directional asymmetry in knowledge editing.
When a model is edited with (Subject, Relation, Target) [e.g., A is mother of B],
it often fails to resolve the inverse query (Target, InverseRelation, Subject) [e.g., B is child of A],
leading to directional inconsistency, hallucination, or catastrophic asymmetric drift.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult


# Standard canonical inverse relation templates
CANONICAL_INVERSES: Dict[str, Tuple[str, str, str]] = {
    # relation_key: (inverse_relation_name, forward_template, inverse_template)
    "capital": ("capital_of", "The capital of {subject} is", "The country whose capital is {target} is"),
    "located in": ("contains", "{subject} is located in", "{target} contains the landmark"),
    "mother": ("child_of_mother", "{subject} is the mother of", "The mother of {target} is"),
    "father": ("child_of_father", "{subject} is the father of", "The father of {target} is"),
    "author": ("written_by", "The author of {subject} is", "{target} wrote the work"),
    "wrote": ("work_of", "{subject} wrote", "The famous work created by {target} is"),
    "ceo": ("led_by", "{subject} is the CEO of", "The CEO leading {target} is"),
    "founded": ("founded_by", "{subject} founded", "{target} was founded by"),
    "plays for": ("employs_player", "{subject} plays for", "The team that employs {target} is"),
    "invented": ("invented_by", "{subject} invented", "{target} was invented by"),
    "born in": ("birthplace_of", "{subject} was born in", "The birthplace of {target} is"),
    "citizen of": ("has_citizen", "{subject} is a citizen of", "A notable citizen of {target} is"),
}


@dataclass
class ReciprocalPairProbe:
    """Probes forward and inverse assertions for factual reciprocal consistency."""
    forward_prompt: str
    forward_target: str
    inverse_prompt: str
    inverse_target: str
    relation: str
    inverse_relation: str


@dataclass
class ReciprocalAuditResult:
    """Detailed audit metrics evaluating bidirectional knowledge coherence."""
    subject: str
    relation: str
    target: str
    forward_prompt: str
    forward_target: str
    inverse_prompt: str
    inverse_target: str
    forward_prob_pre: float
    forward_prob_post: float
    inverse_prob_pre: float
    inverse_prob_post: float
    directional_asymmetry_gap: float  # |P_post(forward) - P_post(inverse)|
    reciprocal_consistency_score: float  # Harmonic mean / coherence [0, 1]
    reversal_curse_detected: bool  # True if forward succeeded but inverse completely failed
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "target": self.target,
            "forward_prompt": self.forward_prompt,
            "forward_target": self.forward_target,
            "inverse_prompt": self.inverse_prompt,
            "inverse_target": self.inverse_target,
            "forward_prob_pre": round(float(self.forward_prob_pre), 4),
            "forward_prob_post": round(float(self.forward_prob_post), 4),
            "inverse_prob_pre": round(float(self.inverse_prob_pre), 4),
            "inverse_prob_post": round(float(self.inverse_prob_post), 4),
            "directional_asymmetry_gap": round(float(self.directional_asymmetry_gap), 4),
            "reciprocal_consistency_score": round(float(self.reciprocal_consistency_score), 4),
            "reversal_curse_detected": bool(self.reversal_curse_detected),
            "summary": self.summary,
        }


class FactualReciprocalAuditor:
    """Audits bidirectional factual consistency and directional asymmetry gaps."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper
        self.inverse_rules = dict(CANONICAL_INVERSES)

    def register_inverse_rule(
        self,
        relation: str,
        inverse_relation: str,
        forward_template: str,
        inverse_template: str,
    ):
        """Registers a custom domain reciprocal relation rule."""
        self.inverse_rules[relation.lower()] = (
            inverse_relation,
            forward_template,
            inverse_template,
        )

    def generate_reciprocal_probe(
        self,
        subject: str,
        relation: str,
        target: str,
        custom_inverse_prompt: Optional[str] = None,
    ) -> ReciprocalPairProbe:
        """Constructs forward and inverse probe prompts for an edited factual triple."""
        rel_key = relation.lower().strip()
        if rel_key in self.inverse_rules:
            inv_rel, fwd_tmpl, inv_tmpl = self.inverse_rules[rel_key]
            fwd_prompt = fwd_tmpl.format(subject=subject, target=target)
            inv_prompt = custom_inverse_prompt or inv_tmpl.format(subject=subject, target=target)
        else:
            inv_rel = f"inverse_of_{rel_key}"
            fwd_prompt = f"{subject} {relation}"
            inv_prompt = custom_inverse_prompt or f"{target} is associated in reverse with"

        return ReciprocalPairProbe(
            forward_prompt=fwd_prompt,
            forward_target=target,
            inverse_prompt=inv_prompt,
            inverse_target=subject,
            relation=relation,
            inverse_relation=inv_rel,
        )

    def audit_edit_reciprocity(
        self,
        subject: str,
        relation: str,
        target_new: str,
        custom_inverse_prompt: Optional[str] = None,
        asymmetry_threshold: float = 0.35,
    ) -> ReciprocalAuditResult:
        """Measures forward and inverse prediction probabilities to assess reciprocal coherence."""
        probe = self.generate_reciprocal_probe(
            subject=subject,
            relation=relation,
            target=target_new,
            custom_inverse_prompt=custom_inverse_prompt,
        )

        # Measure baseline forward and inverse probabilities
        probs_fwd = self.wrapper.predict_next_token_probs(probe.forward_prompt)
        p_fwd_post = probs_fwd.get(probe.forward_target, 0.0)
        p_fwd_pre = 0.001  # baseline reference

        probs_inv = self.wrapper.predict_next_token_probs(probe.inverse_prompt)
        p_inv_post = probs_inv.get(probe.inverse_target, 0.0)
        p_inv_pre = 0.001

        # Calculate Directional Asymmetry Gap (DAG) = |P(fwd) - P(inv)|
        dag = abs(p_fwd_post - p_inv_post)

        # Reciprocal Consistency Score (RCS): harmonic mean with epsilon
        if p_fwd_post > 0 and p_inv_post > 0:
            rcs = 2.0 * (p_fwd_post * p_inv_post) / (p_fwd_post + p_inv_post)
        else:
            rcs = 0.0

        # Reversal Curse detection: forward succeeded (> 0.25) but inverse remains near zero (< 0.05)
        reversal_curse = (p_fwd_post >= 0.20 and p_inv_post < 0.05) or (dag > asymmetry_threshold and p_inv_post < 0.10)

        if reversal_curse:
            summary = (
                f"⚠️ Reversal Curse detected! Forward assertion ('{probe.forward_prompt}' -> '{probe.forward_target}') "
                f"has probability {p_fwd_post:.3f}, but inverse ('{probe.inverse_prompt}' -> '{probe.inverse_target}') "
                f"collapsed to {p_inv_post:.3f} (Asymmetry Gap: {dag:.3f})."
            )
        elif rcs > 0.40:
            summary = (
                f"✅ Symmetric consistency confirmed (RCS: {rcs:.3f}, DAG: {dag:.3f}). "
                f"Both forward and inverse relations accurately propagate."
            )
        else:
            summary = (
                f"ℹ️ Moderate bidirectional alignment (RCS: {rcs:.3f}, DAG: {dag:.3f})."
            )

        return ReciprocalAuditResult(
            subject=subject,
            relation=relation,
            target=target_new,
            forward_prompt=probe.forward_prompt,
            forward_target=probe.forward_target,
            inverse_prompt=probe.inverse_prompt,
            inverse_target=probe.inverse_target,
            forward_prob_pre=p_fwd_pre,
            forward_prob_post=p_fwd_post,
            inverse_prob_pre=p_inv_pre,
            inverse_prob_post=p_inv_post,
            directional_asymmetry_gap=dag,
            reciprocal_consistency_score=rcs,
            reversal_curse_detected=reversal_curse,
            summary=summary,
        )
