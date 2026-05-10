"""FastAPI endpoint tests."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import NUM_CLASSES


class _FakeEngine:
    """Stub inference engine for endpoint tests."""

    def __init__(self):
        self.models: dict[str, object] = {"resnet50": object()}
        self.device = "cpu"
        self.default_model = "resnet50"

    def load_available(self):
        return list(self.models.keys())

    def predict_bytes(self, image_bytes: bytes,
                      model_name: str | None = None):
        from src.api.inference import InferenceOutput
        return InferenceOutput(
            top_k=[("imran_khan", 0.85), ("nawaz_sharif", 0.10),
                   ("shehbaz_sharif", 0.05)],
            face_bbox=(10, 10, 100, 100),
            face_detected=True,
            inference_time_ms=42.0,
        )

    def format_top_k(self, top_k):
        from src.config import get_display_name
        return [{"class": n, "display_name": get_display_name(n),
                 "confidence": round(c, 4)} for n, c in top_k]


@pytest.fixture
def client():
    app = create_app()
    app.state.engine = _FakeEngine()
    with TestClient(app) as c:
        yield c


def _make_image_bytes() -> bytes:
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    arr = np.full((256, 256, 3), 120, dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("healthy", "degraded")
    assert "resnet50" in data["models_loaded"]


def test_classes_endpoint_returns_30(client):
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert data["num_classes"] == NUM_CLASSES
    assert len(data["classes"]) == NUM_CLASSES


def test_predict_with_valid_image(client):
    img = _make_image_bytes()
    r = client.post("/predict",
                     files={"file": ("test.jpg", img, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["class"] == "imran_khan"
    assert data["face_detected"] is True
    assert data["face_bbox"] == [10, 10, 100, 100]


def test_predict_with_non_image_returns_415(client):
    r = client.post("/predict",
                     files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_with_oversized_file_returns_413(client):
    big = b"\xff" * (11 * 1024 * 1024)
    r = client.post("/predict",
                     files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_batch_predict(client):
    img = _make_image_bytes()
    r = client.post("/predict-batch", files=[
        ("files", ("a.jpg", img, "image/jpeg")),
        ("files", ("b.jpg", img, "image/jpeg")),
    ])
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
