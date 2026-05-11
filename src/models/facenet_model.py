"""InceptionResnetV1 pretrained on VGGFace2 — face-specific backbone."""
from __future__ import annotations

from torch import Tensor, nn

from src.config import get_param
from src.models.base_model import BaseClassifier


class FaceNetClassifier(BaseClassifier):
    """VGGFace2-pretrained Inception-ResNet-V1 backbone + custom head."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__(num_classes, pretrained, freeze_backbone)
        from facenet_pytorch import InceptionResnetV1
        self.backbone = InceptionResnetV1(
            pretrained="vggface2" if pretrained else None,
            classify=False,
        )
        dropout = get_param("training", "dropout", 0.3)
        hidden = get_param("training", "hidden_dim", 512)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, hidden),
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
        return "facenet_vggface2"

    def get_backbone_modules(self) -> list[nn.Module]:
        return [self.backbone]

    def get_head(self) -> nn.Module:
        return self.head

    def get_target_layer_for_gradcam(self) -> nn.Module:
        return self.backbone.block8


if __name__ == "__main__":
    import torch
    m = FaceNetClassifier(num_classes=12, pretrained=False)
    x = torch.randn(2, 3, 160, 160)
    print("out shape:", m(x).shape)
