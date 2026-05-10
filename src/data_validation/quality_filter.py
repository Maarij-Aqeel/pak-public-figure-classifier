"""Quality filter: blur, resolution, aspect ratio, file size, color."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import METADATA_DIR, REJECTED_DIR, VALIDATED_DIR, get_param
from src.utils.helpers import append_csv_row, list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)

QUALITY_LOG = METADATA_DIR / "quality_log.csv"
QUALITY_LOG_FIELDS = [
    "path", "width", "height", "blur_score",
    "aspect_ratio", "file_size", "status", "reason",
]


@dataclass
class QualityResult:
    """One image's quality outcome."""
    path: Path
    status: str
    reason: str = ""
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    aspect_ratio: float = 0.0
    file_size: int = 0


class QualityFilter:
    """Reject blurry, low-res, oddly shaped, or icon-sized images."""

    def __init__(self):
        self.min_resolution = get_param("validation", "min_resolution", 224)
        self.blur_threshold = get_param("validation", "blur_threshold", 100.0)
        self.min_file_size = get_param("validation",
                                         "min_file_size_bytes", 5120)
        self.max_aspect_ratio = get_param("validation", "max_aspect_ratio", 3.0)

    def assess(self, path: Path) -> QualityResult:
        """Compute all quality metrics for a single image."""
        try:
            file_size = path.stat().st_size
        except Exception as exc:
            return QualityResult(path, "rejected", f"stat_error:{exc}")

        if file_size < self.min_file_size:
            return QualityResult(path, "rejected", "file_too_small",
                                 file_size=file_size)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                arr = np.array(img)
        except Exception as exc:
            return QualityResult(path, "rejected", f"open_error:{exc}",
                                 file_size=file_size)

        if w < self.min_resolution or h < self.min_resolution:
            return QualityResult(path, "rejected", "low_resolution",
                                 width=w, height=h, file_size=file_size)

        aspect = w / h
        if aspect > self.max_aspect_ratio or aspect < (1 / self.max_aspect_ratio):
            return QualityResult(path, "rejected", "bad_aspect_ratio",
                                 width=w, height=h,
                                 aspect_ratio=round(aspect, 3),
                                 file_size=file_size)

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.blur_threshold:
            return QualityResult(path, "rejected", "blurry",
                                 width=w, height=h, blur_score=blur,
                                 aspect_ratio=round(aspect, 3),
                                 file_size=file_size)

        return QualityResult(path, "kept", "", width=w, height=h,
                             blur_score=blur,
                             aspect_ratio=round(aspect, 3),
                             file_size=file_size)

    def process_class(self, class_name: str) -> dict[str, int]:
        """Run quality filter on all validated images for a class."""
        images = list_images(VALIDATED_DIR / class_name)
        stats = {"total": len(images), "kept": 0, "rejected": 0}
        for img_path in images:
            r = self.assess(img_path)
            if r.status != "kept":
                stats["rejected"] += 1
                self._reject(img_path, r.reason, class_name)
            else:
                stats["kept"] += 1
            append_csv_row(QUALITY_LOG, {
                "path": str(img_path),
                "width": r.width,
                "height": r.height,
                "blur_score": round(r.blur_score, 2),
                "aspect_ratio": r.aspect_ratio,
                "file_size": r.file_size,
                "status": r.status,
                "reason": r.reason,
            }, QUALITY_LOG_FIELDS)
        logger.info("[%s] quality filter: kept=%d rejected=%d",
                    class_name, stats["kept"], stats["rejected"])
        return stats

    def _reject(self, path: Path, reason: str, class_name: str) -> None:
        """Move a rejected image to rejected/{reason}/{class}."""
        dst_dir = REJECTED_DIR / reason / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(dst_dir / path.name))
        except Exception as exc:
            logger.debug("reject move failed %s: %s", path, exc)


if __name__ == "__main__":
    from src.config import ensure_dirs
    ensure_dirs()
    qf = QualityFilter()
    print(f"min_res={qf.min_resolution} blur_thr={qf.blur_threshold}")
