"""Orchestrate face detection -> dedup -> quality -> outliers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import (
    CLASS_NAMES,
    METADATA_DIR,
    RAW_DIR,
    VALIDATED_DIR,
    ensure_dirs,
)
from src.data_validation.cross_class_checker import CrossClassChecker
from src.data_validation.deduplicator import Deduplicator
from src.data_validation.face_detector import FaceDetector
from src.data_validation.quality_filter import QualityFilter
from src.utils.helpers import list_images, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALIDATION_REPORT = METADATA_DIR / "validation_report.json"


@dataclass
class ValidationReport:
    """Final validation summary."""
    total_raw_images: int = 0
    kept_after_face_detection: int = 0
    kept_after_dedup: int = 0
    kept_after_quality: int = 0
    flagged_outliers: int = 0
    per_class_final_counts: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""


def validate_all(skip_outliers: bool = False) -> ValidationReport:
    """Run the full validation pipeline across all classes."""
    ensure_dirs()
    report = ValidationReport(timestamp=datetime.now().isoformat())

    face = FaceDetector()
    dedup = Deduplicator()
    quality = QualityFilter()

    raw_total = sum(len(list_images(RAW_DIR / c)) for c in CLASS_NAMES)
    report.total_raw_images = raw_total
    logger.info("Validation start: %d raw images across %d classes",
                raw_total, len(CLASS_NAMES))

    face_stats_total = 0
    for cls in CLASS_NAMES:
        stats = face.process_class(RAW_DIR, cls)
        face_stats_total += stats["kept"]
    report.kept_after_face_detection = face_stats_total

    dedup_total = 0
    for cls in CLASS_NAMES:
        s = dedup.dedup_within_class(cls)
        dedup_total += s.kept
    report.kept_after_dedup = dedup_total

    quality_total = 0
    for cls in CLASS_NAMES:
        s = quality.process_class(cls)
        quality_total += s["kept"]
    report.kept_after_quality = quality_total

    dedup.detect_cross_class_collisions()

    if not skip_outliers:
        try:
            checker = CrossClassChecker()
            flagged_map = checker.check_all_classes()
            report.flagged_outliers = sum(flagged_map.values())
        except Exception as exc:
            logger.warning("Outlier check skipped: %s", exc)

    report.per_class_final_counts = {
        cls: len(list_images(VALIDATED_DIR / cls)) for cls in CLASS_NAMES
    }
    write_json(VALIDATION_REPORT, asdict(report))

    logger.info("Validation complete: raw=%d → final=%d (outliers flagged=%d)",
                report.total_raw_images,
                sum(report.per_class_final_counts.values()),
                report.flagged_outliers)
    return report


if __name__ == "__main__":
    rep = validate_all(skip_outliers=True)
    print("Per-class final:", rep.per_class_final_counts)
