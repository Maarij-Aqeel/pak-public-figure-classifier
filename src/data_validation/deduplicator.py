"""Perceptual hash deduplication, within and across classes."""
from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from src.config import (
    CLASS_NAMES,
    METADATA_DIR,
    REJECTED_DIR,
    VALIDATED_DIR,
    get_param,
)
from src.utils.helpers import append_csv_row, list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


DEDUP_LOG = METADATA_DIR / "dedup_log.csv"
DEDUP_LOG_FIELDS = ["kept_path", "removed_path", "kept_class",
                     "removed_class", "distance", "scope"]

CROSS_CLASS_LOG = METADATA_DIR / "cross_class_dedup.csv"
CROSS_CLASS_LOG_FIELDS = ["path_a", "class_a", "path_b", "class_b", "distance"]


@dataclass
class DedupStats:
    """Outcome of a dedup pass."""
    total: int = 0
    kept: int = 0
    removed: int = 0
    groups: int = 0


class Deduplicator:
    """Group near-duplicates by perceptual hash distance."""

    def __init__(self):
        self.hash_distance = get_param("validation", "dedup_hash_distance", 5)

    def compute_hash(self, path: Path) -> imagehash.ImageHash | None:
        """Compute pHash for a file."""
        try:
            with Image.open(path) as img:
                return imagehash.phash(img.convert("RGB"))
        except Exception as exc:
            logger.debug("hash fail %s: %s", path, exc)
            return None

    def dedup_within_class(self, class_name: str) -> DedupStats:
        """Remove near-duplicates inside one class folder."""
        class_dir = VALIDATED_DIR / class_name
        images = list_images(class_dir)
        stats = DedupStats(total=len(images))
        if len(images) < 2:
            stats.kept = len(images)
            return stats

        sized = [(p, self._image_pixels(p)) for p in images]
        sized.sort(key=lambda t: t[1], reverse=True)
        hashes: list[tuple[Path, imagehash.ImageHash]] = []
        keepers: list[Path] = []
        removed_groups = 0

        for path, _ in sized:
            h = self.compute_hash(path)
            if h is None:
                continue
            dup_of = None
            for kept_path, kept_hash in hashes:
                if (h - kept_hash) <= self.hash_distance:
                    dup_of = kept_path
                    break
            if dup_of is None:
                hashes.append((path, h))
                keepers.append(path)
            else:
                self._move_duplicate(path, class_name)
                append_csv_row(DEDUP_LOG, {
                    "kept_path": str(dup_of),
                    "removed_path": str(path),
                    "kept_class": class_name,
                    "removed_class": class_name,
                    "distance": int(h - self.compute_hash(dup_of) or 0),
                    "scope": "within",
                }, DEDUP_LOG_FIELDS)
                removed_groups += 1

        stats.kept = len(keepers)
        stats.removed = removed_groups
        stats.groups = len(hashes)
        logger.info("[%s] within-class dedup: kept=%d removed=%d",
                    class_name, stats.kept, stats.removed)
        return stats

    def detect_cross_class_collisions(self) -> int:
        """Flag images that appear across classes (do NOT auto-delete)."""
        all_hashes: dict[str, list[tuple[Path, imagehash.ImageHash]]] = defaultdict(list)
        for cls in CLASS_NAMES:
            for p in list_images(VALIDATED_DIR / cls):
                h = self.compute_hash(p)
                if h is not None:
                    all_hashes[cls].append((p, h))

        collisions = 0
        classes = list(all_hashes.keys())
        for i, cls_a in enumerate(classes):
            for cls_b in classes[i + 1:]:
                for pa, ha in all_hashes[cls_a]:
                    for pb, hb in all_hashes[cls_b]:
                        dist = ha - hb
                        if dist <= self.hash_distance:
                            collisions += 1
                            append_csv_row(CROSS_CLASS_LOG, {
                                "path_a": str(pa),
                                "class_a": cls_a,
                                "path_b": str(pb),
                                "class_b": cls_b,
                                "distance": int(dist),
                            }, CROSS_CLASS_LOG_FIELDS)
        logger.info("Cross-class near-duplicates flagged: %d", collisions)
        return collisions

    def _image_pixels(self, path: Path) -> int:
        """Total pixel count (for keeping highest-res when deduping)."""
        try:
            with Image.open(path) as img:
                w, h = img.size
                return w * h
        except Exception:
            return 0

    def _move_duplicate(self, path: Path, class_name: str) -> None:
        """Move a duplicate to rejected/duplicates."""
        dst_dir = REJECTED_DIR / "duplicates" / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(dst_dir / path.name))
        except Exception as exc:
            logger.debug("move dup failed %s: %s", path, exc)


if __name__ == "__main__":
    from src.config import ensure_dirs
    ensure_dirs()
    dedup = Deduplicator()
    print("Dedup hash distance:", dedup.hash_distance)
