"""MTCNN-based face detection, cropping, and rejection."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    METADATA_DIR,
    REJECTED_DIR,
    VALIDATED_DIR,
    get_param,
)
from src.utils.helpers import append_csv_row, list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


FACE_LOG = METADATA_DIR / "face_detection_log.csv"
FACE_LOG_FIELDS = [
    "original_path", "validated_path", "faces_found",
    "confidence", "bbox", "status", "reason",
]


@dataclass
class FaceResult:
    """One image's detection outcome."""
    original_path: Path
    status: str
    reason: str = ""
    validated_path: Path | None = None
    faces_found: int = 0
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


class FaceDetector:
    """Wraps facenet-pytorch MTCNN with rejection rules."""

    def __init__(self, device: str | None = None):
        self.confidence_threshold = get_param(
            "validation", "face_confidence_threshold", 0.95
        )
        self.min_area_ratio = get_param(
            "validation", "min_face_area_ratio", 0.10
        )
        self.padding_ratio = get_param(
            "validation", "face_padding_ratio", 0.20
        )
        self.device = device
        self._mtcnn = None

    def _lazy_mtcnn(self):
        """Construct MTCNN on first use (heavy import)."""
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            import torch
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._mtcnn = MTCNN(
                keep_all=True,
                device=dev,
                post_process=False,
                select_largest=False,
            )
            logger.info("MTCNN initialized on %s", dev)
        return self._mtcnn

    def detect(self, image: Image.Image) -> FaceResult:
        """Detect faces in a PIL image and return decision."""
        try:
            mtcnn = self._lazy_mtcnn()
            arr = np.asarray(image)
            boxes, probs = mtcnn.detect(arr)
        except Exception as exc:
            return FaceResult(Path(""), "rejected", f"detect_error:{exc}")

        if boxes is None or len(boxes) == 0:
            return FaceResult(Path(""), "rejected", "no_face_detected", faces_found=0)

        faces = list(zip(boxes, probs or []))
        faces = [(b, p) for b, p in faces if p is not None
                 and p >= self.confidence_threshold]
        if not faces:
            return FaceResult(Path(""), "rejected", "low_confidence",
                              faces_found=len(boxes))

        if len(faces) > 1:
            faces.sort(
                key=lambda bp: (bp[0][2] - bp[0][0]) * (bp[0][3] - bp[0][1]),
                reverse=True,
            )

        box, conf = faces[0]
        x1, y1, x2, y2 = box
        w, h = image.size
        face_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        if face_area / (w * h) < self.min_area_ratio:
            return FaceResult(Path(""), "rejected", "face_too_small",
                              faces_found=len(faces),
                              confidence=float(conf),
                              bbox=(int(x1), int(y1), int(x2), int(y2)))

        return FaceResult(
            Path(""), "kept", "",
            faces_found=len(faces),
            confidence=float(conf),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
        )

    def crop_with_padding(self, image: Image.Image,
                          bbox: tuple[int, int, int, int]) -> Image.Image:
        """Crop face region with padding around the bbox."""
        x1, y1, x2, y2 = bbox
        w, h = image.size
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return image.crop((x1, y1, x2, y2))

    def process_image(self, src_path: Path, class_name: str) -> FaceResult:
        """Process a single raw image: detect, crop, save (or reject)."""
        try:
            image = Image.open(src_path).convert("RGB")
        except Exception as exc:
            return FaceResult(src_path, "rejected", f"open_error:{exc}")

        result = self.detect(image)
        result.original_path = src_path

        if result.status != "kept":
            reject_dir = REJECTED_DIR / result.reason / class_name
            reject_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_path, reject_dir / src_path.name)
            except Exception:
                pass
            return result

        out_dir = VALIDATED_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cropped = self.crop_with_padding(image, result.bbox)
        out_path = out_dir / src_path.name
        try:
            cropped.save(out_path, format="JPEG", quality=92)
            result.validated_path = out_path
        except Exception as exc:
            return FaceResult(src_path, "rejected", f"save_error:{exc}")
        return result

    def process_class(self, raw_dir: Path, class_name: str) -> dict[str, int]:
        """Run face detection on every raw image of a class."""
        images = list_images(raw_dir / class_name)
        stats = {"total": len(images), "kept": 0, "rejected": 0}
        for img_path in images:
            res = self.process_image(img_path, class_name)
            if res.status == "kept":
                stats["kept"] += 1
            else:
                stats["rejected"] += 1
            append_csv_row(
                FACE_LOG,
                {
                    "original_path": str(res.original_path),
                    "validated_path": str(res.validated_path or ""),
                    "faces_found": res.faces_found,
                    "confidence": round(res.confidence, 4),
                    "bbox": str(res.bbox or ""),
                    "status": res.status,
                    "reason": res.reason,
                },
                FACE_LOG_FIELDS,
            )
        logger.info("[%s] face detection: kept=%d rejected=%d (of %d)",
                    class_name, stats["kept"], stats["rejected"], stats["total"])
        return stats


if __name__ == "__main__":
    from src.config import ensure_dirs
    ensure_dirs()
    detector = FaceDetector()
    print("FaceDetector configured. Conf threshold:",
          detector.confidence_threshold)
