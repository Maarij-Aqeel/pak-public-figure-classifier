"""Final test-set evaluation: metrics, confusion matrix, misclassified, Grad-CAM."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import MODELS_DIR, NUM_CLASSES, ensure_dirs, get_param
from src.data_preprocessing.dataset import get_dataloader
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.gradcam import generate_gradcam_grid
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.misclassified_analyzer import (
    find_misclassified,
    render_grid,
    save_misclassified_csv,
)
from src.models import build_model
from src.utils.logger import get_logger, timestamped_log_path


def parse_args() -> argparse.Namespace:
    """CLI flags."""
    p = argparse.ArgumentParser(description="Final evaluation")
    p.add_argument("--model", type=str, required=True,
                   help="model name (resnet50|efficientnet_b3|vit_b_16)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to .pt checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu|mps; default=auto")
    p.add_argument("--skip-gradcam", action="store_true")
    return p.parse_args()


def pick_device(arg: str | None) -> str:
    """Auto-pick device if not specified."""
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """Top-level entry."""
    ensure_dirs()
    args = parse_args()
    log_path = timestamped_log_path("evaluation")
    logger = get_logger("run_evaluation", log_file=log_path.name)
    device = pick_device(args.device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        MODELS_DIR / f"{args.model}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    logger.info("Loading %s from %s", args.model, ckpt_path)
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    test_loader = get_dataloader("test", shuffle=False)
    preds, labels, probs = collect_predictions(model, test_loader, device)

    compute_metrics(preds, labels, model_name=args.model)
    plot_confusion_matrix(preds, labels, model_name=args.model)
    mis_rows = find_misclassified(preds, labels, probs, split="test")
    save_misclassified_csv(mis_rows, args.model)
    render_grid(mis_rows, args.model)

    if get_param("evaluation", "generate_gradcam", True) and not args.skip_gradcam:
        try:
            generate_gradcam_grid(model, args.model, device, split="test")
        except Exception as exc:
            logger.warning("Grad-CAM step failed: %s", exc)


if __name__ == "__main__":
    main()
