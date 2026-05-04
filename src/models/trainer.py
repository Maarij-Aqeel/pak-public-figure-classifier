"""Unified trainer with MLflow, AMP, label smoothing, cosine LR."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CLASS_NAMES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RESULTS_DIR,
    get_param,
)
from src.models.base_model import BaseClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Summary returned by Trainer.train()."""
    model_name: str
    best_val_accuracy: float = 0.0
    best_val_f1: float = 0.0
    best_epoch: int = -1
    best_checkpoint_path: str = ""
    final_test_accuracy: float = 0.0
    final_test_f1: float = 0.0
    epochs_run: int = 0
    history: list[dict] = field(default_factory=list)


class Trainer:
    """Two-stage trainer (warmup head, then fine-tune backbone)."""

    def __init__(self, model: BaseClassifier, train_loader: DataLoader,
                 val_loader: DataLoader, device: str | None = None,
                 class_weights: torch.Tensor | None = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.model.to(self.device)

        self.batch_size = get_param("training", "batch_size", 32)
        self.warmup_epochs = get_param("training", "warmup_epochs", 5)
        self.fine_tune_epochs = get_param("training", "fine_tune_epochs", 45)
        self.warmup_lr = get_param("training", "warmup_lr", 1e-3)
        self.fine_tune_lr = get_param("training", "fine_tune_lr", 1e-4)
        self.weight_decay = get_param("training", "weight_decay", 1e-4)
        self.label_smoothing = get_param("training", "label_smoothing", 0.1)
        self.patience = get_param("training", "early_stopping_patience", 8)
        self.grad_clip = get_param("training", "gradient_clip", 1.0)
        self.use_amp = get_param("training", "use_amp", True) \
            and self.device == "cuda"

        weight_tensor = class_weights.to(self.device) \
            if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=self.label_smoothing,
        )
        self.scaler = GradScaler(device=self.device, enabled=self.use_amp)

    def _build_optimizer(self, lr: float) -> optim.Optimizer:
        """AdamW over currently-trainable params."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer: optim.Optimizer,
                          total_epochs: int) -> optim.lr_scheduler.LRScheduler:
        """Cosine annealing over remaining epochs."""
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs, 1)
        )

    def _train_one_epoch(self, optimizer: optim.Optimizer) -> dict[str, float]:
        """Single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(self.train_loader, desc="train", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        return {"loss": running_loss / max(total, 1),
                "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Forward pass over a loader, returning loss/acc/f1."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_preds: list[int] = []
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
        f1 = f1_score(all_labels, all_preds, average="weighted",
                      zero_division=0) if all_labels else 0.0
        return {
            "loss": running_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "f1": f1,
        }

    def train(self, run_name: str | None = None,
              test_loader: DataLoader | None = None) -> TrainingResult:
        """Run two-stage training and return summary."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        except Exception as exc:
            logger.warning("set_experiment failed (%s); using default", exc)

        model_name = self.model.get_model_name()
        run_name = run_name or \
            f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = TrainingResult(model_name=model_name)
        ckpt_dir = MODELS_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / f"{model_name}_best.pt"

        total_epochs = self.warmup_epochs + self.fine_tune_epochs
        epochs_no_improve = 0

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_name": model_name,
                "batch_size": self.batch_size,
                "warmup_epochs": self.warmup_epochs,
                "fine_tune_epochs": self.fine_tune_epochs,
                "warmup_lr": self.warmup_lr,
                "fine_tune_lr": self.fine_tune_lr,
                "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing,
                "grad_clip": self.grad_clip,
                "use_amp": self.use_amp,
                "device": self.device,
                "num_classes": self.model.num_classes,
            })
            trainable, total = self.model.parameter_count()
            mlflow.set_tags({
                "model_type": model_name,
                "total_params": str(total),
                "trainable_params": str(trainable),
            })

            optimizer = self._build_optimizer(self.warmup_lr)
            scheduler = self._build_scheduler(optimizer, self.warmup_epochs)

            for epoch in range(1, total_epochs + 1):
                if epoch == self.warmup_epochs + 1:
                    self.model.unfreeze_top_n_layers(n=2)
                    optimizer = self._build_optimizer(self.fine_tune_lr)
                    scheduler = self._build_scheduler(
                        optimizer, self.fine_tune_epochs
                    )
                    logger.info("Stage 2: unfroze top layers, lr=%.1e",
                                self.fine_tune_lr)

                tr = self._train_one_epoch(optimizer)
                val = self._evaluate(self.val_loader)
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                epoch_log = {
                    "epoch": epoch,
                    "train_loss": tr["loss"],
                    "train_acc": tr["accuracy"],
                    "val_loss": val["loss"],
                    "val_acc": val["accuracy"],
                    "val_f1": val["f1"],
                    "lr": current_lr,
                }
                result.history.append(epoch_log)
                result.epochs_run = epoch
                mlflow.log_metrics({k: v for k, v in epoch_log.items()
                                     if k != "epoch"}, step=epoch)
                logger.info(
                    "ep=%d train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f",
                    epoch, tr["loss"], tr["accuracy"], val["loss"],
                    val["accuracy"], val["f1"]
                )

                if val["accuracy"] > result.best_val_accuracy:
                    result.best_val_accuracy = val["accuracy"]
                    result.best_val_f1 = val["f1"]
                    result.best_epoch = epoch
                    torch.save({
                        "state_dict": self.model.state_dict(),
                        "model_name": model_name,
                        "num_classes": self.model.num_classes,
                        "class_names": CLASS_NAMES,
                        "epoch": epoch,
                        "val_accuracy": val["accuracy"],
                    }, best_path)
                    result.best_checkpoint_path = str(best_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

            history_path = RESULTS_DIR / f"{model_name}_history.json"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(result.history, indent=2))
            mlflow.log_artifact(str(history_path))

            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device,
                                    weights_only=False)
                self.model.load_state_dict(ckpt["state_dict"])
            if test_loader is not None:
                test_metrics = self._evaluate(test_loader)
                result.final_test_accuracy = test_metrics["accuracy"]
                result.final_test_f1 = test_metrics["f1"]
                mlflow.log_metrics({
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1": test_metrics["f1"],
                    "test_loss": test_metrics["loss"],
                })
                logger.info("Test: acc=%.4f f1=%.4f",
                            test_metrics["accuracy"], test_metrics["f1"])

            try:
                mlflow.pytorch.log_model(self.model, "model")
            except Exception as exc:
                logger.warning("mlflow.log_model failed: %s", exc)

            mlflow.log_dict(asdict(result), f"{model_name}_summary.json")

        return result


if __name__ == "__main__":
    print("Trainer module loaded. Use scripts/run_training.py to launch.")
