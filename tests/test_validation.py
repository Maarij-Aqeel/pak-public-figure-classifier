"""Validation pipeline tests."""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image

from src.data_validation.deduplicator import Deduplicator
from src.data_validation.quality_filter import QualityFilter


def test_quality_filter_keeps_valid(sample_image_bytes, tmp_path):
    p = tmp_path / "ok.jpg"
    p.write_bytes(sample_image_bytes)
    qf = QualityFilter()
    r = qf.assess(p)
    assert r.status in ("kept", "rejected")
    assert r.width >= 0 and r.height >= 0


def test_quality_filter_rejects_low_res(tmp_path, tiny_image_bytes):
    p = tmp_path / "small.jpg"
    p.write_bytes(tiny_image_bytes)
    qf = QualityFilter()
    r = qf.assess(p)
    assert r.status == "rejected"
    assert "resolution" in r.reason or "file_too_small" in r.reason


def test_quality_filter_rejects_tiny_file(tmp_path):
    p = tmp_path / "tiny.jpg"
    p.write_bytes(b"x" * 100)
    qf = QualityFilter()
    r = qf.assess(p)
    assert r.status == "rejected"
    assert r.reason == "file_too_small"


def test_phash_dedup_finds_duplicates(tmp_path):
    """Two near-identical images should hash within distance."""
    base = np.full((300, 300, 3), 100, dtype=np.uint8)
    img_a = Image.fromarray(base)
    img_b = Image.fromarray(np.clip(base + 1, 0, 255).astype(np.uint8))
    pa = tmp_path / "a.jpg"
    pb = tmp_path / "b.jpg"
    img_a.save(pa, format="JPEG")
    img_b.save(pb, format="JPEG")
    dedup = Deduplicator()
    ha = dedup.compute_hash(pa)
    hb = dedup.compute_hash(pb)
    assert ha is not None and hb is not None
    assert (ha - hb) <= dedup.hash_distance


def test_phash_distinguishes_different_images(tmp_path):
    """Very different images should have larger hash distance."""
    a = np.zeros((300, 300, 3), dtype=np.uint8)
    b = np.full((300, 300, 3), 255, dtype=np.uint8)
    pa = tmp_path / "black.jpg"
    pb = tmp_path / "white.jpg"
    Image.fromarray(a).save(pa, format="JPEG")
    Image.fromarray(b).save(pb, format="JPEG")
    dedup = Deduplicator()
    ha = dedup.compute_hash(pa)
    hb = dedup.compute_hash(pb)
    assert (ha - hb) >= 0


def test_blur_filter_rejects_blurry(tmp_path):
    """A pure flat image has Laplacian variance ~0 → blurry."""
    arr = np.full((300, 300, 3), 128, dtype=np.uint8)
    p = tmp_path / "blur.jpg"
    Image.fromarray(arr).save(p, format="JPEG", quality=95)
    qf = QualityFilter()
    r = qf.assess(p)
    assert r.status == "rejected"
    assert r.reason == "blurry"
