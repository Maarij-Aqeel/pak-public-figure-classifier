"""Abstract scraper interface — all scrapers inherit BaseScraper."""
from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.config import METADATA_DIR, get_param
from src.utils.helpers import append_csv_row, bytes_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_LOG = METADATA_DIR / "collection_log.csv"
COLLECTION_LOG_FIELDS = [
    "filename", "class", "source", "query", "scraped_date",
    "file_size", "width", "height", "sha256_short",
]


class BaseScraper(ABC):
    """Common scraper interface and helpers."""

    source_name: str = "base"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200):
        self.output_dir = Path(output_dir)
        self.max_images_per_class = max_images_per_class
        self.min_size = get_param("validation", "min_resolution", 224)
        self.min_file_size = get_param("validation", "min_file_size_bytes", 5120)

    @abstractmethod
    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Run scraper for one (query, class) pair; return saved file paths."""

    def _class_dir(self, class_name: str) -> Path:
        d = self.output_dir / class_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_image(self, image_bytes: bytes, class_name: str,
                    query: str = "") -> Path | None:
        """Persist image bytes with conventional name; returns path or None."""
        if not self._is_valid_image(image_bytes):
            return None

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

        w, h = img.size
        sha8 = bytes_hash(image_bytes, "sha256")[:8]
        ts = int(time.time() * 1000)
        filename = f"{class_name}_{self.source_name}_{ts}_{sha8}.jpg"
        out_path = self._class_dir(class_name) / filename

        if out_path.exists():
            return out_path

        try:
            img.save(out_path, format="JPEG", quality=92)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", out_path.name, exc)
            return None

        try:
            append_csv_row(
                COLLECTION_LOG,
                {
                    "filename": filename,
                    "class": class_name,
                    "source": self.source_name,
                    "query": query,
                    "scraped_date": datetime.now().isoformat(),
                    "file_size": out_path.stat().st_size,
                    "width": w,
                    "height": h,
                    "sha256_short": sha8,
                },
                COLLECTION_LOG_FIELDS,
            )
        except Exception as exc:
            logger.warning("Failed to write log row: %s", exc)

        return out_path

    def _is_valid_image(self, image_bytes: bytes) -> bool:
        """Cheap pre-validation: format + min size + min resolution."""
        if not image_bytes or len(image_bytes) < self.min_file_size:
            return False
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
        except Exception:
            return False
        return w >= self.min_size and h >= self.min_size


if __name__ == "__main__":
    print(f"COLLECTION_LOG path: {COLLECTION_LOG}")
    print(f"Fields: {COLLECTION_LOG_FIELDS}")
