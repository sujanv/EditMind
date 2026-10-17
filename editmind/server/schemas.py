"""Pydantic schemas for EditMind REST API."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class EditApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The Eiffel Tower is in"})
    target_new: str = Field(..., json_schema_extra={"example": "Rome"})
    ground_truth: Optional[str] = Field(None, json_schema_extra={"example": "Paris"})
    subject: Optional[str] = Field(None, json_schema_extra={"example": "Eiffel Tower"})
    method: str = Field("rome", json_schema_extra={"example": "rome"})
    model_name: str = Field("toy_causal_lm", json_schema_extra={"example": "toy_causal_lm"})


class TraceApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The Eiffel Tower is in"})
    subject: str = Field(..., json_schema_extra={"example": "Eiffel Tower"})
    target: str = Field(..., json_schema_extra={"example": "Paris"})
    model_name: str = Field("toy_causal_lm", json_schema_extra={"example": "toy_causal_lm"})


class CompareApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The Eiffel Tower is in"})
    target_new: str = Field(..., json_schema_extra={"example": "Rome"})
    ground_truth: Optional[str] = Field(None, json_schema_extra={"example": "Paris"})
    subject: Optional[str] = Field(None, json_schema_extra={"example": "Eiffel Tower"})
    methods: List[str] = Field(default_factory=lambda: ["rome", "memit", "mend", "grace", "ike", "ft_l"])


class MultiHopApiRequest(BaseModel):
    anchor_prompt: str = Field(..., json_schema_extra={"example": "Lionel Messi plays for"})
    anchor_subject: str = Field(..., json_schema_extra={"example": "Lionel Messi"})
    anchor_relation: str = Field(..., json_schema_extra={"example": "plays for"})
    anchor_target: str = Field(..., json_schema_extra={"example": "Inter Miami"})
    downstream_queries: List[Dict[str, Any]] = Field(default_factory=list)
    method: str = Field("grace", json_schema_extra={"example": "grace"})


class AuditApiRequest(BaseModel):
    reference_texts: Optional[List[str]] = Field(None)
    max_allowed_kl: float = Field(1.25, json_schema_extra={"example": 1.25})
    max_allowed_ppl_ratio: float = Field(1.40, json_schema_extra={"example": 1.40})


class RollbackApiRequest(BaseModel):
    patch_id: Optional[str] = Field(None)
    rollback_all: bool = Field(False)


class FactualityApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The Eiffel Tower is in"})
    target_new: str = Field(..., json_schema_extra={"example": "Rome"})
    num_samples: int = Field(5, json_schema_extra={"example": 5})
    temperature: float = Field(0.7, json_schema_extra={"example": 0.7})


class ResolveConflictApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "Lionel Messi plays for"})
    target_new: str = Field(..., json_schema_extra={"example": "Inter Miami"})
    subject: str = Field(..., json_schema_extra={"example": "Lionel Messi"})
    relation: str = Field(..., json_schema_extra={"example": "plays for"})
    strategy: str = Field("latest_wins", json_schema_extra={"example": "latest_wins"})
    valid_from: Optional[str] = Field(None, json_schema_extra={"example": "2023"})
    valid_to: Optional[str] = Field(None)
    confidence: float = Field(1.0, json_schema_extra={"example": 1.0})


class LocalizeRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The capital of France is"})
    subject: str = Field(..., json_schema_extra={"example": "France"})
    target: str = Field(..., json_schema_extra={"example": "Paris"})
    top_k: int = Field(2, json_schema_extra={"example": 2})


class ConceptUnlearnRequest(BaseModel):
    concept_name: str = Field(..., json_schema_extra={"example": "confidential_credentials"})
    forget_prompts: List[str] = Field(..., json_schema_extra={"example": ["The secret key of Alice is", "Alice confidential token was"]})
    retain_prompts: List[str] = Field(..., json_schema_extra={"example": ["The capital of France is Paris", "Shakespeare was an English writer"]})
    target_layers: Optional[List[int]] = Field(None, json_schema_extra={"example": [2, 3]})
