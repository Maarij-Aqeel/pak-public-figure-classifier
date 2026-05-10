"""Scraper interface and helper tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data_collection.base_scraper import BaseScraper
from src.data_collection.icrawler_scraper import build_query_variants


class DummyScraper(BaseScraper):
    source_name = "dummy"

    def scrape(self, query: str, class_name: str):
        return []


def test_base_scraper_is_abstract():
    with pytest.raises(TypeError):
        BaseScraper(Path("/tmp"))


def test_dummy_scraper_instantiation(tmp_path):
    s = DummyScraper(tmp_path, max_images_per_class=10)
    assert s.source_name == "dummy"
    assert s.scrape("q", "c") == []


def test_valid_image_check(tmp_path, sample_image_bytes):
    s = DummyScraper(tmp_path)
    assert s._is_valid_image(sample_image_bytes) is True


def test_invalid_image_rejected_too_small(tmp_path, tiny_image_bytes):
    s = DummyScraper(tmp_path)
    assert s._is_valid_image(tiny_image_bytes) is False


def test_invalid_image_rejected_empty(tmp_path):
    s = DummyScraper(tmp_path)
    assert s._is_valid_image(b"") is False


def test_invalid_image_rejected_garbage(tmp_path):
    s = DummyScraper(tmp_path)
    assert s._is_valid_image(b"not an image" * 1000) is False


def test_save_image_creates_file(tmp_path, sample_image_bytes):
    s = DummyScraper(tmp_path)
    out = s._save_image(sample_image_bytes, "imran_khan", query="test")
    assert out is not None
    assert out.exists()
    assert out.name.startswith("imran_khan_dummy_")
    assert out.suffix == ".jpg"


def test_query_variants_includes_base_name():
    qs = build_query_variants("imran_khan")
    assert any("Imran Khan" == q or q.startswith("Imran Khan") for q in qs)
    assert len(qs) >= 4


def test_query_variants_for_military_class():
    qs = build_query_variants("asim_munir")
    assert any("uniform" in q.lower() for q in qs)
