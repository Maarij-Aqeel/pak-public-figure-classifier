"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceEngine
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    engine = InferenceEngine()
    loaded = engine.load_available()
    if not loaded:
        logger.warning("Starting API with no models loaded.")
    app.state.engine = engine
    yield
    app.state.engine = None


def create_app() -> FastAPI:
    """Construct the FastAPI app."""
    app = FastAPI(
        title="Pakistani Public Figures Classifier",
        description="Image classification API for 30 Pakistani public figures.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
