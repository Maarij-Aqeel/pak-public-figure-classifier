"""PyTorch Dataset and DataLoader helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import CLASS_TO_IDX, SPLITS_DIR, get_param
from src.data_preprocessing.augmentation import build_transform
from src.utils.helpers import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceDataset(Dataset):
    """Image-folder style dataset built on top of CLASS_TO_IDX."""

    def __init__(self, split: str, image_size: int | None = None,
                 split_dir: Path | None = None):
        self.split = split
        self.transform = build_transform(split, image_size)
        self.split_dir = split_dir or (SPLITS_DIR / split)
        self.samples: list[tuple[Path, int]] = self._gather()
        if not self.samples:
            logger.warning("No samples found in %s", self.split_dir)

    def _gather(self) -> list[tuple[Path, int]]:
        """Collect (path, label_idx) pairs."""
        pairs: list[tuple[Path, int]] = []
        for cls, idx in CLASS_TO_IDX.items():
            for p in list_images(self.split_dir / cls):
                pairs.append((p, idx))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load image, apply transform, return tensor + label."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        out = self.transform(image=arr)
        return out["image"], label

    def class_counts(self) -> dict[int, int]:
        """Histogram of labels in this split."""
        c = Counter(label for _, label in self.samples)
        return dict(c)


def get_dataloader(split: str, batch_size: int | None = None,
                   num_workers: int | None = None,
                   shuffle: bool | None = None,
                   image_size: int | None = None) -> DataLoader:
    """Build a configured DataLoader for a split."""
    bs = batch_size or get_param("training", "batch_size", 32)
    nw = num_workers if num_workers is not None else get_param(
        "training", "num_workers", 4
    )
    if shuffle is None:
        shuffle = split == "train"
    ds = FaceDataset(split, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=nw > 0,
    )
    return loader


def get_class_weights(split: str = "train") -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    ds = FaceDataset(split)
    counts = ds.class_counts()
    n_classes = len(CLASS_TO_IDX)
    total = sum(counts.values()) or 1
    weights = torch.ones(n_classes, dtype=torch.float32)
    for idx in range(n_classes):
        c = counts.get(idx, 0)
        weights[idx] = total / (n_classes * max(c, 1))
    return weights


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        try:
            ds = FaceDataset(split)
            print(f"{split}: {len(ds)} samples")
        except Exception as exc:
            print(f"{split}: {exc}")
