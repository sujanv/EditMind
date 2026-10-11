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
