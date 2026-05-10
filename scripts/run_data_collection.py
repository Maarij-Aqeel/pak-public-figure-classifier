"""Master script: scrape images for all (or selected) classes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import (
    CLASS_NAMES,
    METADATA_DIR,
    RAW_DIR,
    ensure_dirs,
    get_param,
)
from src.data_collection.unified_collector import UnifiedCollector
from src.utils.helpers import class_image_counts, list_images, write_json
from src.utils.logger import get_logger, timestamped_log_path

CHECKPOINT_FILE = METADATA_DIR / "collection_checkpoint.json"
REPORT_FILE = METADATA_DIR / "collection_report.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Run image collection")
    parser.add_argument("--classes", type=str, default="",
                        help="comma-separated subset; default=all")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last checkpoint")
    parser.add_argument("--target-count", type=int, default=None,
                        help="target images per class")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip classes already at target")
    parser.add_argument("--use-selenium", action="store_true",
                        help="enable gov-site Selenium fallback")
    return parser.parse_args()


def selected_classes(arg: str) -> list[str]:
    """Resolve --classes flag."""
    if not arg:
        return list(CLASS_NAMES)
    return [c.strip() for c in arg.split(",") if c.strip() in CLASS_NAMES]


def load_checkpoint() -> set[str]:
    """Return set of classes already completed."""
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", []))
    except Exception:
        return set()


def save_checkpoint(completed: set[str]) -> None:
    """Persist checkpoint."""
    write_json(CHECKPOINT_FILE, {"completed": sorted(completed),
                                  "updated_at": datetime.now().isoformat()})


def main() -> None:
    """Top-level entry."""
    ensure_dirs()
    args = parse_args()
    log_path = timestamped_log_path("collection")
    logger = get_logger("run_data_collection", log_file=log_path.name)

    classes = selected_classes(args.classes)
    target = args.target_count or get_param("data_collection",
                                              "target_per_class", 200)
    min_per_class = get_param("data_collection", "min_per_class", 80)

    completed = load_checkpoint() if args.resume else set()

    collector = UnifiedCollector(
        output_dir=RAW_DIR,
        target_per_class=target,
        use_selenium=args.use_selenium,
    )

    logger.info("Collecting %d classes (target=%d, resume=%s)",
                len(classes), target, args.resume)

    all_results: list[dict] = []
    for cls in classes:
        if cls in completed:
            logger.info("[%s] already in checkpoint, skipping", cls)
            continue
        existing = len(list_images(RAW_DIR / cls))
        if args.skip_existing and existing >= target:
            logger.info("[%s] already has %d images, skipping", cls, existing)
            completed.add(cls)
            save_checkpoint(completed)
            continue

        result = collector.collect_class(cls)
        all_results.append({
            "class": cls,
            "total": result.total_collected,
            "per_source": result.per_source_counts,
            "duration_s": round(result.duration_seconds, 2),
            "errors": result.errors,
            "meets_min": result.total_collected >= min_per_class,
        })
        completed.add(cls)
        save_checkpoint(completed)

    counts = class_image_counts(RAW_DIR, CLASS_NAMES)
    total_images = sum(counts.values())
    below_min = [c for c, n in counts.items() if n < min_per_class]

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "per_class_counts": counts,
        "classes_below_min": below_min,
        "per_class_runs": all_results,
        "target_per_class": target,
        "min_per_class": min_per_class,
    }
    write_json(REPORT_FILE, report)
    logger.info("Collection done. Total=%d. Below min: %d classes",
                total_images, len(below_min))
    if below_min:
        logger.warning("Classes below minimum (%d): %s", min_per_class, below_min)


if __name__ == "__main__":
    main()
