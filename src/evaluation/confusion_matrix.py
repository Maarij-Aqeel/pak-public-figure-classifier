"""30x30 confusion matrix heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                          model_name: str = "model",
                          out_path: Path | None = None,
                          sort_by_accuracy: bool = True) -> Path:
    """Render and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    class_names = list(CLASS_NAMES)

    if sort_by_accuracy:
        per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        order = np.argsort(-per_class_acc)
        cm = cm[order][:, order]
        class_names = [CLASS_NAMES[i] for i in order]

    out_path = out_path or RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    annot = np.where(cm > 1, cm.astype(str), "").astype(object)
    sns.heatmap(cm, cmap="Blues", annot=annot, fmt="",
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
    return out_path


if __name__ == "__main__":
    print("Confusion matrix module loaded.")
