"""Albumentations transforms for train and val/test splits."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import get_param

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int | None = None) -> A.Compose:
    """Strong augmentations for the training split."""
    size = image_size or get_param("preprocessing", "image_size", 224)
    resize = get_param("preprocessing", "resize_size", 256)
    return A.Compose([
        A.Resize(resize, resize),
        A.RandomCrop(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                    contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                              sat_shift_limit=15,
                              val_shift_limit=10, p=0.3),
        A.RandomShadow(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20,
                         min_holes=1, min_height=4, min_width=4, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int | None = None) -> A.Compose:
    """Deterministic transforms for val/test/inference."""
    size = image_size or get_param("preprocessing", "image_size", 224)
    resize = get_param("preprocessing", "resize_size", 256)
    return A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_transform(split: str, image_size: int | None = None) -> A.Compose:
    """Dispatch on split name."""
    if split == "train":
        return build_train_transform(image_size)
    return build_eval_transform(image_size)


if __name__ == "__main__":
    t_tr = build_train_transform()
    t_ev = build_eval_transform()
    print("train ops:", len(t_tr.transforms))
    print("eval  ops:", len(t_ev.transforms))
