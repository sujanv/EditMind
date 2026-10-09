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
