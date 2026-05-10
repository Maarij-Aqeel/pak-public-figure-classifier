"""Wikipedia + Wikimedia Commons image scraper."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src.config import get_display_name
from src.data_collection.base_scraper import BaseScraper
from src.utils.logger import get_logger

logger = get_logger(__name__)

USER_AGENT = "pak-public-figures-classifier/1.0 (academic; contact: maintainers)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


class WikipediaScraper(BaseScraper):
    """High-quality images from Wikipedia pages + Wikimedia Commons."""

    source_name = "wikipedia"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200,
                 per_query_limit: int = 20):
        super().__init__(output_dir, max_images_per_class)
        self.per_query_limit = per_query_limit
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Combine page-image and Commons-search results."""
        saved: list[Path] = []
        saved += self._scrape_page_images(query, class_name)
        saved += self._scrape_commons(query, class_name)
        return saved

    def scrape_class(self, class_name: str) -> list[Path]:
        """Use display name as primary query."""
        return self.scrape(get_display_name(class_name), class_name)

    def _scrape_page_images(self, query: str, class_name: str) -> list[Path]:
        """Pull all images on the matching Wikipedia page."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "images",
                "titles": query,
                "imlimit": self.per_query_limit,
            }
            resp = self.session.get(WIKI_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Wiki page lookup failed for '%s': %s", query, exc)
            return []

        pages = data.get("query", {}).get("pages", {})
        image_titles: list[str] = []
        for page in pages.values():
            for img in page.get("images", []):
                title = img.get("title", "")
                if any(skip in title.lower() for skip in
                       ["icon", "commons-logo", "edit-icon", "flag of"]):
                    continue
                image_titles.append(title)

        saved: list[Path] = []
        for title in image_titles[: self.per_query_limit]:
            url = self._image_url(title)
            if not url:
                continue
            data_bytes = self._download(url)
            if not data_bytes:
                continue
            out = self._save_image(data_bytes, class_name, query=f"wiki:{query}")
            if out is not None:
                saved.append(out)
            time.sleep(0.3)
        logger.info("Wikipedia '%s' -> %d images (class=%s)",
                    query, len(saved), class_name)
        return saved

    def _scrape_commons(self, query: str, class_name: str) -> list[Path]:
        """Query Wikimedia Commons image search."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srsearch": query,
                "srlimit": self.per_query_limit,
            }
            resp = self.session.get(COMMONS_API, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
        except Exception as exc:
            logger.warning("Commons search failed for '%s': %s", query, exc)
            return []

        saved: list[Path] = []
        for hit in results:
            title = hit.get("title", "")
            url = self._image_url(title)
            if not url:
                continue
            data_bytes = self._download(url)
            if not data_bytes:
                continue
            out = self._save_image(data_bytes, class_name, query=f"commons:{query}")
            if out is not None:
                saved.append(out)
            time.sleep(0.3)
        logger.info("Commons '%s' -> %d images (class=%s)",
                    query, len(saved), class_name)
        return saved

    def _image_url(self, file_title: str) -> str | None:
        """Resolve File:Foo.jpg to direct URL via imageinfo."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "titles": file_title,
            }
            resp = self.session.get(COMMONS_API, params=params, timeout=10)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [])
                if info:
                    return info[0].get("url")
        except Exception as exc:
            logger.debug("imageinfo failed for %s: %s", file_title, exc)
        return None

    def _download(self, url: str) -> bytes | None:
        """Download bytes with size guard."""
        try:
            resp = self.session.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            content = resp.content
            if len(content) > 25 * 1024 * 1024:
                return None
            return content
        except Exception as exc:
            logger.debug("Download failed %s: %s", url, exc)
            return None


if __name__ == "__main__":
    from src.config import RAW_DIR, ensure_dirs
    ensure_dirs()
    scraper = WikipediaScraper(RAW_DIR, max_images_per_class=10, per_query_limit=5)
    print("Scraper ready:", scraper.source_name)
