"""FastAPI route handlers."""
from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.api.schemas import (
    BatchItem,
    BatchResponse,
    ClassesResponse,
    ClassInfo,
    HealthResponse,
    PredictResponse,
)
from src.config import CLASS_NAMES, get_display_name, get_param
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_FILE_BYTES = get_param("api", "max_file_size_mb", 10) * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp",
}


def _engine(request: Request):
    """Read inference engine from app state."""
    return request.app.state.engine


def _validate_upload(file: UploadFile, content: bytes) -> None:
    """Reject non-image or oversized payloads."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415,
            f"unsupported content type: {file.content_type}")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413,
            f"file too large: {len(content)} > {MAX_FILE_BYTES}")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness."""
    engine = _engine(request)
    return HealthResponse(
        status="healthy" if engine.models else "degraded",
        models_loaded=sorted(engine.models.keys()),
        device=engine.device,
    )


@router.get("/classes", response_model=ClassesResponse)
async def classes() -> ClassesResponse:
    """List all 30 classes."""
    items = [
        ClassInfo(id=i, name=name, display_name=get_display_name(name))
        for i, name in enumerate(CLASS_NAMES)
    ]
    return ClassesResponse(classes=items, num_classes=len(items))


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request,
                  file: UploadFile = File(...),
                  model: str = Query(default="")) -> PredictResponse:
    """Single-image top-K prediction."""
    content = await file.read()
    _validate_upload(file, content)

    engine = _engine(request)
    chosen = model or engine.default_model
    try:
        out = engine.predict_bytes(content, model_name=chosen)
    except Exception as exc:
        logger.exception("predict failed: %s", exc)
        raise HTTPException(500, f"inference error: {exc}")

    if not out.face_detected:
        raise HTTPException(422, "no face detected in uploaded image")

    return PredictResponse(
        predictions=engine.format_top_k(out.top_k),
        model_used=chosen,
        face_detected=out.face_detected,
        face_bbox=list(out.face_bbox) if out.face_bbox else None,
        inference_time_ms=round(out.inference_time_ms, 2),
    )


@router.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(request: Request,
                        files: list[UploadFile] = File(...),
                        model: str = Query(default="")) -> BatchResponse:
    """Multi-image prediction."""
    engine = _engine(request)
    chosen = model or engine.default_model

    t_start = time.perf_counter()
    results: list[BatchItem] = []
    for f in files:
        content = await f.read()
        try:
            _validate_upload(f, content)
        except HTTPException as exc:
            results.append(BatchItem(filename=f.filename or "unknown",
                                      error=exc.detail))
            continue
        try:
            out = engine.predict_bytes(content, model_name=chosen)
            if not out.face_detected:
                results.append(BatchItem(filename=f.filename or "unknown",
                                          error="no face detected"))
                continue
            results.append(BatchItem(
                filename=f.filename or "unknown",
                predictions=[p for p in engine.format_top_k(out.top_k)],
            ))
        except Exception as exc:
            results.append(BatchItem(filename=f.filename or "unknown",
                                      error=str(exc)))

    total_ms = (time.perf_counter() - t_start) * 1000
    return BatchResponse(
        results=results,
        model_used=chosen,
        total_inference_time_ms=round(total_ms, 2),
    )
