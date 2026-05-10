"""Augmentation pipeline tests."""
from __future__ import annotations

import numpy as np

from src.data_preprocessing.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def _arr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)


def test_train_transform_changes_output_between_calls():
    """Random augmentations should differ across calls."""
    img = _arr()
    t = build_train_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    assert a.shape == b.shape
    assert not np.allclose(a.numpy(), b.numpy())


def test_eval_transform_is_deterministic():
    """Eval transform should produce identical output."""
    img = _arr()
    t = build_eval_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    assert np.allclose(a.numpy(), b.numpy())


def test_transform_output_shape():
    """Output should be (3, H, W) and normalized."""
    img = _arr()
    out = build_eval_transform()(image=img)["image"]
    assert out.shape[0] == 3
    assert out.shape[1] == out.shape[2]
    assert out.dtype.is_floating_point


def test_normalization_uses_imagenet_stats():
    """Verify the constants we publish."""
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)
