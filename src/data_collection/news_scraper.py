"""Pakistani news sites image scraper (Dawn, Geo, ARY, etc.)."""
from __future__ import annotations

import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.config import get_display_name
from src.data_collection.base_scraper import BaseScraper
from src.utils.logger import get_logger

logger = get_logger(__name__)


NEWS_SITES = [
    {"name": "dawn",     "search": "https://www.dawn.com/search?q={q}"},
    {"name": "geo",      "search": "https://www.geo.tv/search?q={q}"},
    {"name": "ary",      "search": "https://arynews.tv/?s={q}"},
    {"name": "tribune",  "search": "https://tribune.com.pk/?s={q}"},
    {"name": "thenews",  "search": "https://www.thenews.com.pk/search/{q}"},
    {"name": "samaa",    "search": "https://www.samaa.tv/?s={q}"},
    {"name": "bbcurdu",  "search": "https://www.bbc.com/urdu/search?q={q}"},
]

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

REQUEST_DELAY = 2.0
MAX_ARTICLES_PER_SITE = 8


class NewsScraper(BaseScraper):
    """Search Pakistani news outlets and pull main article images."""

    source_name = "news"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200,
                 per_site_limit: int = 5):
        super().__init__(output_dir, max_images_per_class)
        self.per_site_limit = per_site_limit
        self.session = requests.Session()

    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Walk each news site for a query."""
        saved: list[Path] = []
        for site in NEWS_SITES:
            try:
                saved += self._scrape_site(site, query, class_name)
            except Exception as exc:
                logger.warning("News site %s failed: %s", site["name"], exc)
            time.sleep(REQUEST_DELAY)
        return saved

    def scrape_class(self, class_name: str) -> list[Path]:
        return self.scrape(get_display_name(class_name), class_name)

    def _scrape_site(self, site: dict[str, str], query: str,
                     class_name: str) -> list[Path]:
        """Run search on one site, parse article images."""
        url = site["search"].format(q=requests.utils.quote(query))
        html = self._get(url)
        if not html:
            return []

        article_urls = self._extract_article_urls(html, url)[:MAX_ARTICLES_PER_SITE]
        saved: list[Path] = []
        for art_url in article_urls:
            if len(saved) >= self.per_site_limit:
                break
            art_html = self._get(art_url)
            if not art_html:
                continue
            for img_url in self._extract_image_urls(art_html, art_url):
                data = self._download(img_url)
                if not data:
                    continue
                out = self._save_image(
                    data, class_name,
                    query=f"{site['name']}:{query}",
                )
                if out is not None:
                    saved.append(out)
                    break
            time.sleep(REQUEST_DELAY)
        logger.info("News '%s' [%s] -> %d images (class=%s)",
                    query, site["name"], len(saved), class_name)
        return saved

    def _get(self, url: str) -> str | None:
        """GET with rotating UA, gentle on errors."""
        headers = {"User-Agent": random.choice(UA_LIST), "Accept": "text/html"}
        try:
            r = self.session.get(url, headers=headers, timeout=15)
            if r.status_code in (403, 429):
                logger.debug("Blocked %d: %s", r.status_code, url)
                return None
            r.raise_for_status()
            return r.text
        except Exception as exc:
            logger.debug("GET failed %s: %s", url, exc)
            return None

    def _extract_article_urls(self, html: str, base_url: str) -> list[str]:
        """Pull plausible article links from a search results page."""
        soup = BeautifulSoup(html, "html.parser")
        base = urlparse(base_url)
        seen: set[str] = set()
        urls: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#") or href.startswith("javascript:"):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc != base.netloc:
                continue
            if full in seen:
                continue
            if not any(seg in parsed.path for seg in
                       ["/news/", "/article", "/story", "/202", "/politics", "/pakistan"]):
                continue
            seen.add(full)
            urls.append(full)
        return urls

    def _extract_image_urls(self, html: str, page_url: str) -> list[str]:
        """og:image first, then large content imgs."""
        soup = BeautifulSoup(html, "html.parser")
        urls: list[str] = []

        og = soup.find("meta", attrs={"property": "og:image"})
        if og and og.get("content"):
            urls.append(urljoin(page_url, og["content"]))

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if not src or src.startswith("data:"):
                continue
            full = urljoin(page_url, src)
            if full not in urls:
                urls.append(full)
        return urls

    def _download(self, url: str) -> bytes | None:
        """Fetch image bytes with size guard."""
        headers = {"User-Agent": random.choice(UA_LIST)}
        try:
            r = self.session.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            if len(r.content) > 15 * 1024 * 1024:
                return None
            return r.content
        except Exception as exc:
            logger.debug("Img download failed %s: %s", url, exc)
            return None


if __name__ == "__main__":
    from src.config import RAW_DIR, ensure_dirs
    ensure_dirs()
    scraper = NewsScraper(RAW_DIR, max_images_per_class=10)
    print("News sites configured:", [s["name"] for s in NEWS_SITES])
