"""Dynamic Knowledge Router and Semantic Paradigm Dispatcher.

Analyzes input prompts, identifies semantic domains (e.g. Geography, Science, Pop Culture,
Privacy/PII, Coding), and routes knowledge updates or queries to the mathematically optimal
editing paradigm (e.g. ROME for single-fact MLP edits, AlphaEdit for dense null-space locality,
GRACE for activation codebooks, Steer for non-destructive overrides, CA-Edit for attention heads).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from editmind.core.registry import EditorRegistry
from editmind.models.model_wrapper import UnifiedModelWrapper


# Default domain signatures with representative keywords and affinity mappings
DEFAULT_DOMAINS: Dict[str, Dict[str, Any]] = {
    "geography": {
        "keywords": ["capital", "located", "city", "country", "river", "mountain", "tower", "borders", "island"],
        "recommended_paradigm": "rome",
        "description": "Factual landmark, entity location, and geographical knowledge",
    },
    "science_tech": {
        "keywords": ["invented", "discovered", "theory", "orbit", "planet", "physics", "element", "molecule", "algorithm", "software"],
        "recommended_paradigm": "alphaedit",
        "description": "Scientific discoveries, physical laws, and technical facts requiring null-space preservation",
    },
    "privacy_pii": {
        "keywords": ["password", "secret", "passcode", "ssn", "token", "credential", "private", "confidential", "email", "phone"],
        "recommended_paradigm": "grace",
        "description": "Sensitive credentials, PII erasure, and localized memory interception",
    },
    "transient_override": {
        "keywords": ["hypothetically", "temporarily", "suppose", "imagine", "scenario", "assume", "roleplay"],
        "recommended_paradigm": "steer",
        "description": "Reversible, zero-weight modification steering for hypothetical reasoning",
    },
    "general_factual": {
        "keywords": ["author", "wrote", "born", "actor", "directed", "founded", "ceo", "plays", "won", "award"],
        "recommended_paradigm": "pmet",
        "description": "General biographic, cultural, and corporate factual knowledge",
    },
}


@dataclass
class DomainClassificationResult:
    """Outcome of domain classification and paradigm routing."""
    prompt: str
    detected_domain: str
    confidence: float
    domain_scores: Dict[str, float]
    recommended_paradigm: str
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "detected_domain": self.detected_domain,
            "confidence": round(float(self.confidence), 4),
            "domain_scores": {k: round(float(v), 4) for k, v in self.domain_scores.items()},
            "recommended_paradigm": self.recommended_paradigm,
            "justification": self.justification,
        }


@dataclass
class RouterMetrics:
    """Historical telemetry of router dispatches across domains."""
    total_dispatches: int
    domain_counts: Dict[str, int]
    paradigm_counts: Dict[str, int]
    average_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dispatches": self.total_dispatches,
            "domain_counts": self.domain_counts,
            "paradigm_counts": self.paradigm_counts,
            "average_confidence": round(float(self.average_confidence), 4),
        }


class KnowledgeDomainRouter:
    """Analyzes prompts and routes knowledge edits to optimal editing paradigms."""

    def __init__(self, custom_domains: Optional[Dict[str, Dict[str, Any]]] = None):
        self.domains = dict(DEFAULT_DOMAINS)
        if custom_domains:
            self.domains.update(custom_domains)

        self._history: List[DomainClassificationResult] = []

    def register_domain(
        self,
        domain_name: str,
        keywords: List[str],
        recommended_paradigm: str,
        description: str = "",
    ):
        """Registers a custom semantic domain and its recommended editing paradigm."""
        self.domains[domain_name.lower()] = {
            "keywords": [k.lower() for k in keywords],
            "recommended_paradigm": recommended_paradigm.lower(),
            "description": description,
        }

    def route_prompt(self, prompt: str) -> DomainClassificationResult:
        """Determines domain classification and recommends the optimal editor paradigm."""
        lower_prompt = prompt.lower()
        words = set(lower_prompt.replace(".", " ").replace(",", " ").replace("?", " ").split())

        scores: Dict[str, float] = {}
        for domain, info in self.domains.items():
            kw_list = info["keywords"]
            matches = sum(1 for kw in kw_list if kw in words or any(kw in w for w in words))
            # Calculate affinity score normalized by keyword set length
            score = (matches / max(1, len(kw_list))) * 2.5 + (0.1 if matches > 0 else 0.0)
            scores[domain] = float(min(1.0, score))

        # If no explicit keyword matches, fallback to general_factual with baseline confidence
        if max(scores.values()) == 0.0:
            scores["general_factual"] = 0.35
            detected_domain = "general_factual"
            confidence = 0.35
        else:
            detected_domain = max(scores, key=scores.get)
            confidence = scores[detected_domain]

        recommended_paradigm = self.domains[detected_domain]["recommended_paradigm"]
        desc = self.domains[detected_domain]["description"]
        justification = (
            f"Classified into '{detected_domain}' domain with confidence {confidence * 100:.1f}%. "
            f"Recommending '{recommended_paradigm.upper()}' ({desc})."
        )

        res = DomainClassificationResult(
            prompt=prompt,
            detected_domain=detected_domain,
            confidence=confidence,
            domain_scores=scores,
            recommended_paradigm=recommended_paradigm,
            justification=justification,
        )
        self._history.append(res)
        return res

    def get_metrics(self) -> RouterMetrics:
        """Computes aggregate routing metrics across all past dispatches."""
        total = len(self._history)
        if total == 0:
            return RouterMetrics(
                total_dispatches=0,
                domain_counts={},
                paradigm_counts={},
                average_confidence=0.0,
            )

        dom_counts: Dict[str, int] = {}
        par_counts: Dict[str, int] = {}
        conf_sum = 0.0

        for r in self._history:
            dom_counts[r.detected_domain] = dom_counts.get(r.detected_domain, 0) + 1
            par_counts[r.recommended_paradigm] = par_counts.get(r.recommended_paradigm, 0) + 1
            conf_sum += r.confidence

        return RouterMetrics(
            total_dispatches=total,
            domain_counts=dom_counts,
            paradigm_counts=par_counts,
            average_confidence=conf_sum / total,
        )
