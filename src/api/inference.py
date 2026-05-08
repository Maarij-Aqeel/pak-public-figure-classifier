"""Model loading + single-image inference for the API."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    get_display_name,
    get_param,
)
from src.data_preprocessing.augmentation import build_eval_transform
from src.models import build_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceOutput:
    """Single-image prediction result."""
    top_k: list[tuple[str, float]]
    face_bbox: tuple[int, int, int, int] | None
    face_detected: bool
    inference_time_ms: float


def _pick_device() -> str:
    """Auto-pick inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InferenceEngine:
    """Loads one or more checkpoints and runs predictions."""

    def __init__(self, model_dir: Path | None = None,
                 device: str | None = None):
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.device = device or _pick_device()
        self.models: dict[str, torch.nn.Module] = {}
        self.transform = build_eval_transform()
        self._mtcnn = None
        self.top_k = get_param("evaluation", "top_k", 3)
        self.default_model = get_param("api", "default_model", "resnet50")

    def load_available(self) -> list[str]:
        """Discover and load all *_best.pt checkpoints."""
        loaded: list[str] = []
        if not self.model_dir.exists():
            logger.warning("Model dir missing: %s", self.model_dir)
            return loaded
        for ckpt in sorted(self.model_dir.glob("*_best.pt")):
            model_name = ckpt.stem.replace("_best", "")
            try:
                model = build_model(model_name, num_classes=NUM_CLASSES,
                                     pretrained=False)
                state = torch.load(ckpt, map_location=self.device,
                                    weights_only=False)
                model.load_state_dict(state["state_dict"])
                model.eval().to(self.device)
                self.models[model_name] = model
                loaded.append(model_name)
                logger.info("Loaded model: %s", model_name)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", ckpt, exc)
        return loaded

    def _lazy_mtcnn(self):
        """Construct MTCNN lazily."""
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(keep_all=False, device=self.device,
                                 post_process=False, select_largest=True)
        return self._mtcnn

    def detect_face(self, image: Image.Image) -> tuple[bool, tuple[int, int, int, int] | None]:
        """Return (found, bbox) using MTCNN."""
        try:
            mtcnn = self._lazy_mtcnn()
            boxes, _ = mtcnn.detect(np.asarray(image))
            if boxes is None or len(boxes) == 0:
                return False, None
            x1, y1, x2, y2 = boxes[0]
            return True, (int(x1), int(y1), int(x2), int(y2))
        except Exception as exc:
            logger.debug("face detect fail: %s", exc)
            return False, None

    def preprocess(self, image: Image.Image,
                   bbox: tuple[int, int, int, int] | None) -> torch.Tensor:
        """Crop to face (if available) + apply eval transform."""
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = image.size
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
            image = image.crop((x1, y1, x2, y2))
        arr = np.asarray(image.convert("RGB"))
        t = self.transform(image=arr)["image"]
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image,
                    model_name: str | None = None) -> InferenceOutput:
        """Run end-to-end prediction on a PIL image."""
        chosen = model_name or self.default_model
        if chosen not in self.models:
            if not self.models:
                raise RuntimeError("No models loaded")
            chosen = next(iter(self.models))
            logger.warning("Falling back to model: %s", chosen)
        model = self.models[chosen]

        t0 = time.perf_counter()
        face_found, bbox = self.detect_face(image)
        tensor = self.preprocess(image, bbox)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = probs.argsort()[::-1][: self.top_k]
        top_k = [(CLASS_NAMES[int(i)], float(probs[i])) for i in top_idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InferenceOutput(top_k=top_k, face_bbox=bbox,
                               face_detected=face_found,
                               inference_time_ms=elapsed_ms)

    def predict_bytes(self, image_bytes: bytes,
                      model_name: str | None = None) -> InferenceOutput:
        """Decode raw bytes and predict."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict_pil(image, model_name)

    def format_top_k(self, top_k: list[tuple[str, float]]) -> list[dict]:
        """Shape top-k for response schema."""
        return [
            {
                "class": name,
                "display_name": get_display_name(name),
                "confidence": round(conf, 4),
            } for name, conf in top_k
        ]


if __name__ == "__main__":
    engine = InferenceEngine()
    print("Models found:", engine.load_available())
