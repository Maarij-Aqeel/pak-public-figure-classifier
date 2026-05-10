"""Model architectures + unified trainer."""
from src.models.efficientnet_model import EfficientNetClassifier
from src.models.resnet_model import ResNetClassifier
from src.models.vit_model import ViTClassifier

__all__ = ["ResNetClassifier", "EfficientNetClassifier", "ViTClassifier", "build_model"]


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """Factory by model name."""
    name = name.lower()
    if name == "resnet50":
        return ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "efficientnet_b3":
        return EfficientNetClassifier(num_classes=num_classes,
                                       pretrained=pretrained)
    if name == "vit_b_16":
        return ViTClassifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
