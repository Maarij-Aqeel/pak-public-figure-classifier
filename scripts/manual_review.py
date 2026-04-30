"""Interactive review of borderline images (outliers, multi-face crops)."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from src.config import METADATA_DIR, REJECTED_DIR, VALIDATED_DIR, ensure_dirs
from src.utils.helpers import read_json, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTLIERS_LOG = METADATA_DIR / "outliers.csv"
REVIEW_STATE = METADATA_DIR / "manual_review_state.json"


def parse_args() -> argparse.Namespace:
    """CLI flags."""
    p = argparse.ArgumentParser(description="Manual review CLI")
    p.add_argument("--resume", action="store_true",
                   help="continue from last reviewed index")
    p.add_argument("--limit", type=int, default=0,
                   help="max items to review this session")
    return p.parse_args()


def load_state() -> dict:
    """Load saved progress."""
    if REVIEW_STATE.exists():
        try:
            return read_json(REVIEW_STATE)
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    """Persist progress."""
    write_json(REVIEW_STATE, state)


def load_outliers() -> list[dict[str, str]]:
    """Read outliers CSV."""
    if not OUTLIERS_LOG.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(OUTLIERS_LOG, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "outlier":
                rows.append(row)
    return rows


def display_image(path: Path) -> None:
    """Try PIL show; fallback to printing metadata."""
    try:
        from PIL import Image
        Image.open(path).show()
    except Exception as exc:
        logger.warning("display failed: %s", exc)


def reject_image(path: Path, class_name: str) -> None:
    """Move to rejected/manual_review."""
    dst = REJECTED_DIR / "manual_review" / class_name
    dst.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), str(dst / path.name))
        logger.info("rejected: %s", path.name)
    except Exception as exc:
        logger.warning("move failed: %s", exc)


def main() -> None:
    """Interactive review loop."""
    ensure_dirs()
    args = parse_args()
    state = load_state() if args.resume else {"index": 0, "decisions": {}}
    rows = load_outliers()
    if not rows:
        print("No outliers logged yet. Run validation first.")
        return

    start_idx = state.get("index", 0)
    reviewed = 0
    for i in range(start_idx, len(rows)):
        if args.limit and reviewed >= args.limit:
            break
        row = rows[i]
        path = Path(row["path"])
        if not path.exists():
            continue
        cls = row["class"]
        sim = row.get("similarity_to_centroid", "?")
        print(f"\n[{i + 1}/{len(rows)}] class={cls} similarity={sim}")
        print(f"  path={path}")
        display_image(path)
        choice = input("  [k]eep / [r]eject / [s]kip / [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "r":
            reject_image(path, cls)
            state["decisions"][str(path)] = "rejected"
        elif choice == "k":
            state["decisions"][str(path)] = "kept"
        else:
            state["decisions"][str(path)] = "skipped"
        state["index"] = i + 1
        save_state(state)
        reviewed += 1

    print(f"\nReviewed {reviewed} items. Progress saved.")


if __name__ == "__main__":
    main()
