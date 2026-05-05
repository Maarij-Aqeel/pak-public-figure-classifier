"""EfficientNet-B3 with custom classification head."""
from __future__ import annotations

from torch import Tensor, nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

from src.config import get_param
from src.models.base_model import BaseClassifier


class EfficientNetClassifier(BaseClassifier):
    """EfficientNet-B3 backbone + dropout MLP head."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__(num_classes, pretrained, freeze_backbone)
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        dropout = max(get_param("training", "dropout", 0.3) + 0.1, 0.4)
        hidden = get_param("training", "hidden_dim", 512)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        if freeze_backbone:
            self.freeze_all_backbone()

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def get_model_name(self) -> str:
        return "efficientnet_b3"

    def get_backbone_modules(self) -> list[nn.Module]:
        return list(self.backbone.features)

    def get_head(self) -> nn.Module:
        return self.head

    def get_target_layer_for_gradcam(self) -> nn.Module:
        return self.backbone.features[-1]


if __name__ == "__main__":
    import torch
    m = EfficientNetClassifier(num_classes=30, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("out shape:", m(x).shape)
    print("trainable / total:", m.parameter_count())
