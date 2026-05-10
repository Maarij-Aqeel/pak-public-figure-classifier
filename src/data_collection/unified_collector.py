"""Orchestrator: runs all scrapers per class."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import MILITARY_CLASSES, RAW_DIR, get_param
from src.data_collection.duckduckgo_scraper import DuckDuckGoScraper
from src.data_collection.icrawler_scraper import IcrawlerScraper
from src.data_collection.news_scraper import NewsScraper
from src.data_collection.selenium_fallback import SeleniumScraper
from src.data_collection.wikipedia_scraper import WikipediaScraper
from src.utils.helpers import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CollectionResult:
    """Per-class collection statistics."""
    class_name: str
    total_collected: int = 0
    per_source_counts: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class UnifiedCollector:
    """Wires all scrapers together."""

    def __init__(self, output_dir: Path = RAW_DIR,
                 target_per_class: int | None = None,
                 use_selenium: bool = False):
        self.output_dir = Path(output_dir)
        self.target_per_class = (
            target_per_class
            if target_per_class is not None
            else get_param("data_collection", "target_per_class", 200)
        )
        self.use_selenium = use_selenium

        self.icrawler = IcrawlerScraper(output_dir, self.target_per_class,
                                         per_query_limit=60)
        self.ddg = DuckDuckGoScraper(output_dir, self.target_per_class,
                                      per_query_limit=50)
        self.wiki = WikipediaScraper(output_dir, self.target_per_class,
                                      per_query_limit=15)
        self.news = NewsScraper(output_dir, self.target_per_class,
                                 per_site_limit=5)
        self.selenium = SeleniumScraper(output_dir, self.target_per_class) \
            if use_selenium else None

    def collect_class(self, class_name: str) -> CollectionResult:
        """Run all applicable scrapers for one class."""
        logger.info("Starting collection for class=%s", class_name)
        result = CollectionResult(class_name=class_name)
        start = time.perf_counter()

        for source_name, runner in self._scraper_runners(class_name):
            before = len(list_images(self.output_dir / class_name))
            try:
                runner()
            except Exception as exc:
                msg = f"{source_name} failed: {exc}"
                logger.warning("[%s] %s", class_name, msg)
                result.errors.append(msg)
            after = len(list_images(self.output_dir / class_name))
            result.per_source_counts[source_name] = max(after - before, 0)
            if after >= self.target_per_class:
                logger.info("[%s] target reached at source=%s", class_name, source_name)
                break
            time.sleep(get_param("data_collection", "rate_limit_seconds", 5))

        result.total_collected = len(list_images(self.output_dir / class_name))
        result.duration_seconds = time.perf_counter() - start
        logger.info("[%s] complete: %d images in %.1fs",
                    class_name, result.total_collected, result.duration_seconds)
        return result

    def _scraper_runners(self, class_name: str) -> list[tuple[str, callable]]:
        """Build ordered list of (source_name, callable) to run for this class."""
        runners: list[tuple[str, callable]] = [
            ("icrawler",   lambda: self.icrawler.scrape_class(class_name)),
            ("duckduckgo", lambda: self.ddg.scrape_class(class_name)),
            ("wikipedia",  lambda: self.wiki.scrape_class(class_name)),
            ("news",       lambda: self.news.scrape_class(class_name)),
        ]
        if self.use_selenium and (class_name in MILITARY_CLASSES
                                  or "khan" in class_name):
            runners.append(("selenium",
                            lambda: self.selenium.scrape_class(class_name)))
        return runners


if __name__ == "__main__":
    from src.config import ensure_dirs
    ensure_dirs()
    collector = UnifiedCollector(target_per_class=20)
    print("Collector ready. Target per class:", collector.target_per_class)
