"""Stratified train/val/test split with reproducible seed."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    SPLITS_DIR,
    VALIDATED_DIR,
    ensure_dirs,
    get_param,
)
from src.utils.helpers import list_images, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"


def _clear_splits() -> None:
    """Wipe existing split folders."""
    for split in ("train", "val", "test"):
        d = SPLITS_DIR / split
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _collect_samples() -> tuple[list[Path], list[str]]:
    """Gather (path, class_name) pairs across all classes."""
    paths: list[Path] = []
    labels: list[str] = []
    for cls in CLASS_NAMES:
        for p in list_images(VALIDATED_DIR / cls):
            paths.append(p)
            labels.append(cls)
    return paths, labels


def _copy_to_split(paths: list[Path], labels: list[str], split: str) -> None:
    """Copy files to data/splits/{split}/{class}/ ."""
    for p, lbl in zip(paths, labels):
        dst_dir = SPLITS_DIR / split / lbl
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, dst_dir / p.name)
        except Exception as exc:
            logger.warning("copy fail %s -> %s: %s", p, dst_dir, exc)


def split_dataset(force: bool = False) -> dict:
    """Produce 75/15/10 stratified splits."""
    ensure_dirs()
    train_ratio = get_param("preprocessing", "train_ratio", 0.75)
    val_ratio = get_param("preprocessing", "val_ratio", 0.15)
    test_ratio = get_param("preprocessing", "test_ratio", 0.10)
    seed = get_param("preprocessing", "random_seed", 42)
    eps = 1e-6
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > eps:
        raise ValueError("Split ratios must sum to 1.0")

    if force or any((SPLITS_DIR / s).exists() for s in ("train", "val", "test")):
        _clear_splits()

    paths, labels = _collect_samples()
    if not paths:
        raise RuntimeError("No validated images found to split.")
    logger.info("Splitting %d images across %d classes",
                len(paths), len(set(labels)))

    trainval_p, test_p, trainval_l, test_l = train_test_split(
        paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_p, val_p, train_l, val_l = train_test_split(
        trainval_p, trainval_l,
        test_size=relative_val,
        stratify=trainval_l,
        random_state=seed,
    )

    _copy_to_split(train_p, train_l, "train")
    _copy_to_split(val_p, val_l, "val")
    _copy_to_split(test_p, test_l, "test")

    per_class = {cls: {} for cls in CLASS_NAMES}
    for split, lbls in (("train", train_l), ("val", val_l), ("test", test_l)):
        for cls in CLASS_NAMES:
            per_class[cls][split] = lbls.count(cls)

    manifest = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_samples": len(paths),
        "split_counts": {"train": len(train_l), "val": len(val_l),
                          "test": len(test_l)},
        "per_class": per_class,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(MANIFEST_PATH, manifest)
    logger.info("Split done. train=%d val=%d test=%d",
                len(train_l), len(val_l), len(test_l))
    return manifest


if __name__ == "__main__":
    info = split_dataset(force=True)
    print("Split counts:", info["split_counts"])
