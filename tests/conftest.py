"""Shared pytest fixtures."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_image(size: tuple[int, int] = (256, 256),
                color: tuple[int, int, int] = (128, 64, 32)) -> Image.Image:
    """Solid-color RGB image."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return Image.fromarray(arr)


@pytest.fixture
def sample_image() -> Image.Image:
    """PIL image used by tests."""
    return _make_image()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """JPEG-encoded bytes of a small image."""
    buf = io.BytesIO()
    _make_image((256, 256)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def tiny_image_bytes() -> bytes:
    """Sub-resolution image bytes (should be rejected)."""
    buf = io.BytesIO()
    _make_image((64, 64)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def tmp_class_dir(tmp_path: Path) -> Path:
    """Temp dir mimicking validated/class structure."""
    d = tmp_path / "imran_khan"
    d.mkdir()
    for i in range(3):
        img = _make_image((300, 300), color=(50 + i * 20, 50, 50))
        img.save(d / f"sample_{i}.jpg", format="JPEG")
    return d
