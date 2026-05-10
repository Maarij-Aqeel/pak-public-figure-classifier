"""Grad-CAM visualizations for model interpretability."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import CLASS_NAMES, RESULTS_DIR, SPLITS_DIR
from src.data_preprocessing.augmentation import build_eval_transform
from src.utils.helpers import list_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Apply eval transform and add batch dim."""
    arr = np.array(image)
    t = build_eval_transform()(image=arr)["image"]
    return t.unsqueeze(0)


def generate_gradcam_grid(model, model_name: str, device: str,
                          split: str = "test") -> Path | None:
    """Generate one Grad-CAM heatmap per class."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        logger.warning("grad-cam library not installed; skipping")
        return None

    if not hasattr(model, "get_target_layer_for_gradcam"):
        logger.warning("Model lacks Grad-CAM target layer; skipping")
        return None
    target_layer = model.get_target_layer_for_gradcam()
    model.eval().to(device)
    cam = GradCAM(model=model, target_layers=[target_layer])

    fig, axes = plt.subplots(6, 5, figsize=(14, 16))
    axes = axes.flatten()

    rendered = 0
    for i, cls in enumerate(CLASS_NAMES):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.axis("off")
        imgs = list_images(SPLITS_DIR / split / cls)
        if not imgs:
            continue
        try:
            pil = Image.open(imgs[0]).convert("RGB").resize((224, 224))
            input_t = _to_tensor(pil).to(device)
            mask = cam(input_tensor=input_t)[0]
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            vis = show_cam_on_image(arr, mask, use_rgb=True)
            ax.imshow(vis)
            ax.set_title(cls[:18], fontsize=8)
            rendered += 1
        except Exception as exc:
            logger.debug("gradcam %s failed: %s", cls, exc)

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = RESULTS_DIR / f"{model_name}_gradcam_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Grad-CAM grid saved (%d classes): %s", rendered, out_path)
    return out_path


if __name__ == "__main__":
    print("Grad-CAM module loaded.")
