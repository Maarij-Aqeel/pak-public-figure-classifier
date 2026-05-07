"""Find and visualize top-K misclassified samples per class."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    RESULTS_DIR,
    SPLITS_DIR,
    get_param,
)
from src.utils.helpers import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_misclassified(preds: np.ndarray, labels: np.ndarray,
                       probs: np.ndarray,
                       split: str = "test") -> list[dict]:
    """Build per-sample misclassification rows ordered by class then confidence."""
    paths: list[Path] = []
    for cls in CLASS_NAMES:
        paths.extend(list_images(SPLITS_DIR / split / cls))
    if len(paths) != len(preds):
        logger.warning("path/pred mismatch: %d paths, %d preds — "
                       "filename order may not match. Limiting to min.",
                       len(paths), len(preds))
    n = min(len(paths), len(preds))
    rows = []
    for i in range(n):
        if preds[i] != labels[i]:
            true_idx = int(labels[i])
            pred_idx = int(preds[i])
            true_prob = float(probs[i, true_idx])
            pred_prob = float(probs[i, pred_idx])
            rows.append({
                "path": str(paths[i]),
                "true_class": CLASS_NAMES[true_idx],
                "pred_class": CLASS_NAMES[pred_idx],
                "true_prob": true_prob,
                "pred_prob": pred_prob,
            })
    rows.sort(key=lambda r: r["true_prob"])
    return rows


def save_misclassified_csv(rows: list[dict], model_name: str) -> Path:
    """Write CSV log of misclassifications."""
    out = RESULTS_DIR / f"{model_name}_misclassified.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "true_class", "pred_class", "true_prob", "pred_prob"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    logger.info("Misclassified log: %s (%d rows)", out, len(rows))
    return out


def render_grid(rows: list[dict], model_name: str,
                top_k: int | None = None) -> Path:
    """Render up to K worst misclassifications per class."""
    k = top_k or get_param("evaluation", "max_misclassified_per_class", 5)
    by_class: dict[str, list[dict]] = {cls: [] for cls in CLASS_NAMES}
    for r in rows:
        if len(by_class[r["true_class"]]) < k:
            by_class[r["true_class"]].append(r)

    fig, axes = plt.subplots(NUM_CLASSES, k, figsize=(k * 2.2, NUM_CLASSES * 2))
    if NUM_CLASSES == 1:
        axes = np.array([axes])
    for i, cls in enumerate(CLASS_NAMES):
        for j in range(k):
            ax = axes[i, j] if k > 1 else axes[i]
            ax.axis("off")
            items = by_class.get(cls, [])
            if j >= len(items):
                continue
            r = items[j]
            try:
                img = Image.open(r["path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                continue
            ax.set_title(
                f"T:{cls[:10]}\nP:{r['pred_class'][:10]}\n"
                f"p={r['pred_prob']:.2f}",
                fontsize=7,
            )

    plt.suptitle(f"Top-{k} misclassified per class — {model_name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = RESULTS_DIR / f"{model_name}_misclassified_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Misclassified grid saved: %s", out_path)
    return out_path


if __name__ == "__main__":
    print("Misclassified analyzer loaded.")
