"""Selenium-based fallback for government / JS-heavy sites."""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from src.config import MILITARY_CLASSES, get_display_name
from src.data_collection.base_scraper import BaseScraper
from src.utils.logger import get_logger

logger = get_logger(__name__)


GOV_SITES = [
    "https://www.pmo.gov.pk/",
    "https://na.gov.pk/",
    "https://www.ispr.gov.pk/",
    "https://pid.gov.pk/",
]

PAGE_LOAD_TIMEOUT = 25
SCRAPE_BUDGET_SECONDS = 90


class SeleniumScraper(BaseScraper):
    """Headless Chrome for sites that block requests."""

    source_name = "selenium"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200):
        super().__init__(output_dir, max_images_per_class)

    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Visit gov sites, pull images mentioning the figure."""
        if class_name not in MILITARY_CLASSES and "khan" not in class_name:
            return []
        return self._scrape_with_selenium(query, class_name)

    def scrape_class(self, class_name: str) -> list[Path]:
        return self.scrape(get_display_name(class_name), class_name)

    def _scrape_with_selenium(self, query: str, class_name: str) -> list[Path]:
        """Drive headless Chrome through gov sites."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
        except ImportError:
            logger.warning("Selenium not installed; skipping fallback scraper")
            return []

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        try:
            driver = webdriver.Chrome(options=options)
        except Exception as exc:
            logger.warning("Chromedriver unavailable: %s", exc)
            return []

        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        saved: list[Path] = []
        deadline = time.time() + SCRAPE_BUDGET_SECONDS

        try:
            for site in GOV_SITES:
                if time.time() > deadline:
                    break
                try:
                    driver.get(site)
                    time.sleep(2)
                    imgs = driver.find_elements(By.TAG_NAME, "img")
                    page_text = driver.page_source.lower()
                    if query.lower() not in page_text:
                        continue
                    for el in imgs[:25]:
                        if time.time() > deadline:
                            break
                        src = el.get_attribute("src") or ""
                        if not src or src.startswith("data:"):
                            continue
                        full = urljoin(site, src)
                        data = self._download(full)
                        if not data:
                            continue
                        out = self._save_image(data, class_name,
                                               query=f"gov:{query}")
                        if out is not None:
                            saved.append(out)
                except Exception as exc:
                    logger.debug("Selenium page %s failed: %s", site, exc)
        finally:
            driver.quit()

        logger.info("Selenium '%s' -> %d images (class=%s)",
                    query, len(saved), class_name)
        return saved

    def _download(self, url: str) -> bytes | None:
        """Plain requests download."""
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            if len(r.content) > 15 * 1024 * 1024:
                return None
            return r.content
        except Exception:
            return None


if __name__ == "__main__":
    from src.config import RAW_DIR, ensure_dirs
    ensure_dirs()
    scraper = SeleniumScraper(RAW_DIR)
    print("Selenium scraper ready, target gov sites:", GOV_SITES)
