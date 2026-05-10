"""DuckDuckGo image search scraper — less rate-limited than Google."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src.config import get_display_name
from src.data_collection.base_scraper import BaseScraper
from src.data_collection.icrawler_scraper import build_query_variants
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuckDuckGoScraper(BaseScraper):
    """Wraps duckduckgo_search.DDGS for image discovery."""

    source_name = "duckduckgo"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200,
                 per_query_limit: int = 50):
        super().__init__(output_dir, max_images_per_class)
        self.per_query_limit = per_query_limit
        self.session = requests.Session()

    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Run DDG image search for one query."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo_search not installed; skipping")
            return []

        saved: list[Path] = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    query,
                    max_results=self.per_query_limit,
                    safesearch="moderate",
                    size="Medium",
                ))
        except Exception as exc:
            logger.warning("DDG search failed for '%s': %s", query, exc)
            return []

        for hit in results:
            url = hit.get("image") or hit.get("url")
            if not url:
                continue
            data = self._download(url)
            if not data:
                continue
            out = self._save_image(data, class_name, query=f"ddg:{query}")
            if out is not None:
                saved.append(out)
            time.sleep(0.2)
        logger.info("DDG '%s' -> %d images (class=%s)",
                    query, len(saved), class_name)
        return saved

    def scrape_class(self, class_name: str) -> list[Path]:
        """Run two top query variants."""
        saved: list[Path] = []
        for query in build_query_variants(class_name)[:2]:
            try:
                saved += self.scrape(query, class_name)
            except Exception as exc:
                logger.warning("DDG variant failed (%s): %s", query, exc)
        return saved

    def _download(self, url: str) -> bytes | None:
        """Plain GET with timeout + size guard."""
        try:
            r = self.session.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
            })
            r.raise_for_status()
            if len(r.content) > 15 * 1024 * 1024:
                return None
            return r.content
        except Exception as exc:
            logger.debug("DDG dl failed %s: %s", url, exc)
            return None


if __name__ == "__main__":
    from src.config import RAW_DIR, ensure_dirs
    ensure_dirs()
    scraper = DuckDuckGoScraper(RAW_DIR, max_images_per_class=5, per_query_limit=5)
    print("DDG scraper ready, queries for imran_khan:",
          build_query_variants("imran_khan")[:2])
