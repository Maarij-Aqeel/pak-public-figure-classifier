"""Generic helpers used across modules."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_hash(path: Path, algo: str = "md5") -> str:
    """Hash file contents."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_hash(data: bytes, algo: str = "md5") -> str:
    """Hash a byte string."""
    return hashlib.new(algo, data).hexdigest()


def is_image_file(path: Path) -> bool:
    """Quick extension check."""
    return path.suffix.lower() in VALID_IMAGE_EXTS


def safe_open_image(path: Path) -> Image.Image | None:
    """Open image with PIL or return None on error."""
    try:
        img = Image.open(path)
        img.verify()
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def list_images(directory: Path) -> list[Path]:
    """List image files in a directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and is_image_file(p))


def list_images_recursive(directory: Path) -> list[Path]:
    """List image files recursively."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and is_image_file(p))


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append a row to a CSV, creating header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write json with parents created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def read_json(path: Path) -> Any:
    """Read json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager that yields elapsed time."""
    result: dict[str, float] = {"start": time.perf_counter()}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - result["start"]


def class_image_counts(root: Path, class_names: list[str]) -> dict[str, int]:
    """Count images per class folder."""
    counts: dict[str, int] = {}
    for cls in class_names:
        counts[cls] = len(list_images(root / cls))
    return counts


if __name__ == "__main__":
    print("VALID_IMAGE_EXTS:", VALID_IMAGE_EXTS)
    with timer() as t:
        time.sleep(0.05)
    print("Timer elapsed:", t["elapsed"])
