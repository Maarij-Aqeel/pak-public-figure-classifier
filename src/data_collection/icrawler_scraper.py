"""Google + Bing image scrapers via icrawler library."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.config import MILITARY_CLASSES, get_display_name
from src.data_collection.base_scraper import BaseScraper
from src.utils.helpers import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_query_variants(class_name: str) -> list[str]:
    """Build search query variants for a class."""
    name = get_display_name(class_name)
    queries = [
        name,
        f"{name} politician",
        f"{name} portrait",
        f"{name} press conference",
    ]
    if class_name in MILITARY_CLASSES:
        queries += [f"{name} uniform", f"{name} general"]
    else:
        queries.append(f"{name} parliament")
    return queries


class IcrawlerScraper(BaseScraper):
    """Bundles GoogleImageCrawler and BingImageCrawler."""

    source_name = "icrawler"

    def __init__(self, output_dir: Path, max_images_per_class: int = 200,
                 per_query_limit: int = 60):
        super().__init__(output_dir, max_images_per_class)
        self.per_query_limit = per_query_limit

    def scrape(self, query: str, class_name: str) -> list[Path]:
        """Scrape Google + Bing for a single query."""
        saved: list[Path] = []
        for engine in ("google", "bing"):
            saved += self._run_engine(engine, query, class_name)
        return saved

    def scrape_class(self, class_name: str) -> list[Path]:
        """Run all query variants for a class through both engines."""
        all_saved: list[Path] = []
        for query in build_query_variants(class_name):
            try:
                all_saved += self.scrape(query, class_name)
            except Exception as exc:
                logger.warning("icrawler query failed (%s, %s): %s",
                               class_name, query, exc)
            if len(list_images(self._class_dir(class_name))) >= self.max_images_per_class:
                break
        return all_saved

    def _run_engine(self, engine: str, query: str, class_name: str) -> list[Path]:
        """Crawl one engine to a temp dir, then move valid images into class dir."""
        try:
            from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
        except ImportError:
            logger.error("icrawler not installed; skipping")
            return []

        tmpdir = Path(tempfile.mkdtemp(prefix=f"icrawler_{engine}_"))
        try:
            crawler_cls = GoogleImageCrawler if engine == "google" else BingImageCrawler
            crawler = crawler_cls(
                storage={"root_dir": str(tmpdir)},
                downloader_threads=4,
                feeder_threads=1,
                parser_threads=2,
                log_level=40,
            )
            try:
                crawler.crawl(
                    keyword=query,
                    max_num=self.per_query_limit,
                    min_size=(self.min_size, self.min_size),
                    filters={"type": "photo", "size": "medium"},
                )
            except Exception as exc:
                logger.warning("%s crawl failed for '%s': %s", engine, query, exc)

            saved: list[Path] = []
            for img_path in tmpdir.iterdir():
                if not img_path.is_file():
                    continue
                try:
                    data = img_path.read_bytes()
                except Exception:
                    continue
                out = self._save_image(data, class_name, query=query)
                if out is not None:
                    saved.append(out)
            logger.info("%s '%s' -> %d images (class=%s)",
                        engine, query, len(saved), class_name)
            return saved
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    from src.config import RAW_DIR, ensure_dirs
    ensure_dirs()
    scraper = IcrawlerScraper(RAW_DIR, max_images_per_class=10, per_query_limit=5)
    print("Query variants for imran_khan:")
    print(build_query_variants("imran_khan"))
