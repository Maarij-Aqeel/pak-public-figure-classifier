"""Abstract base classifier."""
from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn


class BaseClassifier(nn.Module):
    """Common interface for all backbones."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Short model identifier."""

    @abstractmethod
    def get_backbone_modules(self) -> list[nn.Module]:
        """Ordered list of backbone modules (head excluded)."""

    @abstractmethod
    def get_head(self) -> nn.Module:
        """Classification head."""

    def freeze_all_backbone(self) -> None:
        """Set requires_grad=False on all backbone params."""
        for module in self.get_backbone_modules():
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_top_n_layers(self, n: int = 1) -> None:
        """Unfreeze the last n backbone modules + head."""
        backbone = self.get_backbone_modules()
        for module in backbone[-n:]:
            for p in module.parameters():
                p.requires_grad = True
        for p in self.get_head().parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All params with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> tuple[int, int]:
        """(trainable, total)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


if __name__ == "__main__":
    print("BaseClassifier loaded:", BaseClassifier.__name__)
    print("torch version:", torch.__version__)
