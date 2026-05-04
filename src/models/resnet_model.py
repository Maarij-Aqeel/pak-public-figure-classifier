"""ResNet-50 with custom classification head."""
from __future__ import annotations

from torch import Tensor, nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.config import get_param
from src.models.base_model import BaseClassifier


class ResNetClassifier(BaseClassifier):
    """ResNet-50 backbone + dropout MLP head."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__(num_classes, pretrained, freeze_backbone)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        dropout = get_param("training", "dropout", 0.3)
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
        return "resnet50"

    def get_backbone_modules(self) -> list[nn.Module]:
        b = self.backbone
        return [
            nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool),
            b.layer1, b.layer2, b.layer3, b.layer4,
        ]

    def get_head(self) -> nn.Module:
        return self.head

    def get_target_layer_for_gradcam(self) -> nn.Module:
        return self.backbone.layer4[-1]


if __name__ == "__main__":
    import torch
    m = ResNetClassifier(num_classes=30, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("out shape:", m(x).shape)
    print("trainable / total:", m.parameter_count())
