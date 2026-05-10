"""CLI wrapper around split_dataset."""
from __future__ import annotations

import argparse

from src.data_preprocessing.splitter import split_dataset
from src.utils.logger import get_logger, timestamped_log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val/test split")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing splits")
    args = parser.parse_args()

    log_path = timestamped_log_path("split")
    logger = get_logger("run_split", log_file=log_path.name)
    manifest = split_dataset(force=args.force)
    logger.info("Split done: %s", manifest["split_counts"])


if __name__ == "__main__":
    main()
