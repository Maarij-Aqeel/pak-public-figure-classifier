"""Overall + per-class precision/recall/F1 metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASS_NAMES, RESULTS_DIR
from src.utils.helpers import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader,
                         device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, labels, probabilities)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    for imgs, labels in tqdm(loader, desc="predict", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs)
    return (np.array(all_preds), np.array(all_labels),
            np.concatenate(all_probs, axis=0))


def compute_metrics(preds: np.ndarray, labels: np.ndarray,
                    model_name: str = "model") -> dict:
    """Compute aggregate and per-class metrics."""
    overall_acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_p = precision_score(labels, preds, average="macro", zero_division=0)
    macro_r = recall_score(labels, preds, average="macro", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    text_report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    per_class = {}
    for cls_name in CLASS_NAMES:
        if cls_name in report:
            per_class[cls_name] = {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"]),
            }

    out = {
        "model": model_name,
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "per_class": per_class,
    }
    write_json(RESULTS_DIR / f"{model_name}_metrics.json", out)
    (RESULTS_DIR / f"{model_name}_classification_report.txt").write_text(text_report)
    logger.info("[%s] acc=%.4f macroF1=%.4f weightedF1=%.4f",
                model_name, overall_acc, macro_f1, weighted_f1)
    return out


if __name__ == "__main__":
    print("Evaluation metrics module loaded.")
