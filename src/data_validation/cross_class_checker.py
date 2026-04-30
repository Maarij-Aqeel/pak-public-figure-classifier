"""FaceNet embedding-based outlier detection per class."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, METADATA_DIR, VALIDATED_DIR, get_param
from src.utils.helpers import append_csv_row, list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


OUTLIERS_LOG = METADATA_DIR / "outliers.csv"
OUTLIERS_LOG_FIELDS = ["path", "class", "similarity_to_centroid", "status"]


@dataclass
class OutlierResult:
    """One image's outlier check."""
    path: Path
    class_name: str
    similarity: float
    is_outlier: bool


class CrossClassChecker:
    """Detect images that don't match the class face embedding centroid."""

    def __init__(self, device: str | None = None):
        self.similarity_threshold = get_param(
            "validation", "outlier_similarity_threshold", 0.5
        )
        self.embedding_samples = get_param(
            "validation", "embedding_samples_per_class", 50
        )
        self.device = device
        self._mtcnn = None
        self._resnet = None

    def _lazy_models(self) -> None:
        """Construct MTCNN + InceptionResnetV1 once."""
        if self._mtcnn is not None:
            return
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(image_size=160, margin=0, post_process=True,
                            device=dev, select_largest=True)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
        self._device = dev
        logger.info("Embedding models loaded on %s", dev)

    def embed(self, path: Path) -> np.ndarray | None:
        """Extract a 512-dim face embedding."""
        self._lazy_models()
        try:
            import torch
            img = Image.open(path).convert("RGB")
            face = self._mtcnn(img)
            if face is None:
                return None
            with torch.no_grad():
                emb = self._resnet(face.unsqueeze(0).to(self._device))
            v = emb.cpu().numpy().flatten()
            n = np.linalg.norm(v)
            return v / n if n > 0 else None
        except Exception as exc:
            logger.debug("embed fail %s: %s", path, exc)
            return None

    def class_centroid(self, class_name: str) -> np.ndarray | None:
        """Compute mean embedding from first N images of a class."""
        imgs = list_images(VALIDATED_DIR / class_name)[: self.embedding_samples]
        vecs: list[np.ndarray] = []
        for p in imgs:
            v = self.embed(p)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        c = np.mean(vecs, axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else None

    def check_class(self, class_name: str) -> list[OutlierResult]:
        """Flag images whose embedding is far from class centroid."""
        centroid = self.class_centroid(class_name)
        if centroid is None:
            logger.warning("[%s] no centroid (no embeddings found)", class_name)
            return []

        results: list[OutlierResult] = []
        for p in list_images(VALIDATED_DIR / class_name):
            v = self.embed(p)
            if v is None:
                continue
            sim = float(np.dot(centroid, v))
            is_outlier = sim < self.similarity_threshold
            results.append(OutlierResult(p, class_name, sim, is_outlier))
            append_csv_row(OUTLIERS_LOG, {
                "path": str(p),
                "class": class_name,
                "similarity_to_centroid": round(sim, 4),
                "status": "outlier" if is_outlier else "ok",
            }, OUTLIERS_LOG_FIELDS)

        outliers = [r for r in results if r.is_outlier]
        logger.info("[%s] outliers flagged: %d / %d",
                    class_name, len(outliers), len(results))
        return results

    def check_all_classes(self) -> dict[str, int]:
        """Run check on every class."""
        flagged: dict[str, int] = {}
        for cls in CLASS_NAMES:
            results = self.check_class(cls)
            flagged[cls] = sum(1 for r in results if r.is_outlier)
        return flagged


if __name__ == "__main__":
    from src.config import ensure_dirs
    ensure_dirs()
    checker = CrossClassChecker()
    print("Outlier threshold:", checker.similarity_threshold)
