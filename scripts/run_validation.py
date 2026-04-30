"""Run full validation pipeline."""
from __future__ import annotations

import argparse

from src.data_validation.validator_pipeline import validate_all
from src.utils.logger import get_logger, timestamped_log_path


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description="Validate scraped images")
    parser.add_argument("--skip-outliers", action="store_true",
                        help="skip FaceNet outlier check (faster)")
    args = parser.parse_args()

    log_path = timestamped_log_path("validation")
    logger = get_logger("run_validation", log_file=log_path.name)

    report = validate_all(skip_outliers=args.skip_outliers)
    logger.info("Final per-class counts:")
    for cls, n in sorted(report.per_class_final_counts.items()):
        logger.info("  %s: %d", cls, n)


if __name__ == "__main__":
    main()
