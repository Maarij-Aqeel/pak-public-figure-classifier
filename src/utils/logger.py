"""Project-wide logger factory."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return configured logger; attaches console + optional file handler once."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def timestamped_log_path(prefix: str) -> Path:
    """Build a timestamped log file path under logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{prefix}_{stamp}.log"


if __name__ == "__main__":
    log = get_logger("logger_test", log_file="logger_test.log")
    log.info("logger works")
    log.warning("warning works")
