"""Metrics computation for knowledge editing evaluation."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from editmind.core.types import EditRequest, EvaluationMetrics
from editmind.models.model_wrapper import UnifiedModelWrapper


def compute_efficacy_score(wrapper: UnifiedModelWrapper, req: EditRequest) -> float:
    """Measures whether P(target_new) > P(ground_truth) on the primary prompt."""
    probs = wrapper.predict_next_token_probs(req.prompt)
    p_new = probs.get(req.target_new, 0.0)
    p_old = probs.get(req.ground_truth, 0.0) if req.ground_truth else 0.0
    return 1.0 if p_new > p_old else 0.0


def compute_generality_score(wrapper: UnifiedModelWrapper, req: EditRequest) -> float:
    """Measures accuracy on paraphrase/rephrase prompts."""
    if not req.rephrase_prompts:
        return 1.0
    successes = 0
    for p in req.rephrase_prompts:
        probs = wrapper.predict_next_token_probs(p)
        p_new = probs.get(req.target_new, 0.0)
        p_old = probs.get(req.ground_truth, 0.0) if req.ground_truth else 0.0
        if p_new >= p_old:
            successes += 1
    return successes / len(req.rephrase_prompts)


def compute_locality_score(wrapper: UnifiedModelWrapper, req: EditRequest) -> float:
    """Measures retention of unrelated neighborhood knowledge."""
    if not req.locality_prompts:
        return 1.0
    retained = 0
    for item in req.locality_prompts:
        prompt = item["prompt"]
        target = item["target"]
        probs = wrapper.predict_next_token_probs(prompt)
        # Check if intended target is among top predictions or has highest prob
        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_tokens = [t[0] for t in sorted_tokens[:3]]
        if target in top_tokens or probs.get(target, 0.0) > 0.01:
            retained += 1
    return retained / len(req.locality_prompts)


def compute_portability_score(wrapper: UnifiedModelWrapper, req: EditRequest) -> float:
    """Measures logical derivation / one-hop compositional deduction."""
    if not req.portability_prompts:
        return 1.0
    passed = 0
    for item in req.portability_prompts:
        prompt = item["prompt"]
        target = item["target"]
        probs = wrapper.predict_next_token_probs(prompt)
        if probs.get(target, 0.0) > 0.005:
            passed += 1
    return passed / len(req.portability_prompts)
