"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """One top-k prediction entry."""
    class_name: str = Field(..., alias="class")
    display_name: str
    confidence: float

    class Config:
        populate_by_name = True


class PredictResponse(BaseModel):
    """Response for /predict."""
    predictions: list[Prediction]
    model_used: str
    face_detected: bool
    face_bbox: list[int] | None = None
    inference_time_ms: float


class BatchItem(BaseModel):
    """One entry in batch response."""
    filename: str
    predictions: list[Prediction] | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Response for /predict-batch."""
    results: list[BatchItem]
    model_used: str
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    models_loaded: list[str]
    device: str


class ClassInfo(BaseModel):
    """One class entry in /classes."""
    id: int
    name: str
    display_name: str


class ClassesResponse(BaseModel):
    """Response for /classes."""
    classes: list[ClassInfo]
    num_classes: int
