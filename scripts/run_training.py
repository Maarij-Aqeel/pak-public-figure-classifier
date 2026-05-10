"""Train all configured models, log to MLflow, generate comparison artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import NUM_CLASSES, RESULTS_DIR, ensure_dirs, get_param
from src.data_preprocessing.dataset import get_class_weights, get_dataloader
from src.models import build_model
from src.models.trainer import Trainer, TrainingResult
from src.utils.helpers import write_json
from src.utils.logger import get_logger, timestamped_log_path


def parse_args() -> argparse.Namespace:
    """CLI flags."""
    p = argparse.ArgumentParser(description="Train classification models")
    p.add_argument("--models", type=str, default="",
                   help="comma-separated subset; default=from params.yaml")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu|mps; default=auto")
    return p.parse_args()


def selected_models(arg: str) -> list[str]:
    """Resolve --models flag."""
    if arg:
        return [m.strip() for m in arg.split(",") if m.strip()]
    return list(get_param("training", "models", ["resnet50", "efficientnet_b3"]))


def plot_training_curves(history: list[dict], model_name: str,
                          out_path: Path) -> None:
    """Plot train/val loss and acc."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title(f"{model_name} loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title(f"{model_name} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def comparison_outputs(results: list[TrainingResult]) -> None:
    """Write comparison CSV + bar chart."""
    if not results:
        return
    df = pd.DataFrame([{
        "model": r.model_name,
        "best_val_accuracy": r.best_val_accuracy,
        "best_val_f1": r.best_val_f1,
        "best_epoch": r.best_epoch,
        "test_accuracy": r.final_test_accuracy,
        "test_f1": r.final_test_f1,
        "epochs_run": r.epochs_run,
    } for r in results])
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["best_val_accuracy"],
           width, label="val_acc")
    ax.bar([i + width / 2 for i in x], df["test_accuracy"],
           width, label="test_acc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Model comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_bar.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Train all models sequentially."""
    ensure_dirs()
    args = parse_args()
    log_path = timestamped_log_path("training")
    logger = get_logger("run_training", log_file=log_path.name)

    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val", shuffle=False)
    test_loader = get_dataloader("test", shuffle=False)
    class_weights = get_class_weights("train")

    results: list[TrainingResult] = []
    for model_name in selected_models(args.models):
        logger.info("=== Training %s ===", model_name)
        model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
        trainer = Trainer(model, train_loader, val_loader,
                          device=args.device, class_weights=class_weights)
        result = trainer.train(test_loader=test_loader)
        results.append(result)
        plot_training_curves(
            result.history, model_name,
            RESULTS_DIR / f"{model_name}_training_curves.png",
        )
        write_json(RESULTS_DIR / f"{model_name}_summary.json",
                   asdict(result))

    comparison_outputs(results)
    if results:
        best = max(results, key=lambda r: r.final_test_accuracy)
        logger.info("Best model: %s @ test_acc=%.4f",
                    best.model_name, best.final_test_accuracy)
        if best.final_test_accuracy < 0.90:
            logger.warning("Below 90%% target. Suggestions: collect more data, "
                           "try ViT, enable TTA, or ensemble models.")
    write_json(RESULTS_DIR / "training_run.json",
               {"timestamp": datetime.now().isoformat(),
                "results": [asdict(r) for r in results]})


if __name__ == "__main__":
    main()
