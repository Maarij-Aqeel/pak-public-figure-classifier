"""Model architecture and trainer sanity tests."""
from __future__ import annotations

import pytest
import torch

from src.config import NUM_CLASSES
from src.models import build_model
from src.models.efficientnet_model import EfficientNetClassifier
from src.models.resnet_model import ResNetClassifier


@pytest.fixture(scope="module")
def dummy_batch() -> torch.Tensor:
    return torch.randn(2, 3, 224, 224)


def test_resnet_output_shape(dummy_batch):
    m = ResNetClassifier(num_classes=NUM_CLASSES, pretrained=False)
    out = m(dummy_batch)
    assert out.shape == (2, NUM_CLASSES)


def test_efficientnet_output_shape(dummy_batch):
    m = EfficientNetClassifier(num_classes=NUM_CLASSES, pretrained=False)
    out = m(dummy_batch)
    assert out.shape == (2, NUM_CLASSES)


def test_build_model_factory(dummy_batch):
    for name in ("resnet50", "efficientnet_b3"):
        m = build_model(name, num_classes=NUM_CLASSES, pretrained=False)
        assert m(dummy_batch).shape == (2, NUM_CLASSES)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_model", num_classes=NUM_CLASSES,
                    pretrained=False)


def test_freeze_backbone_works():
    m = ResNetClassifier(num_classes=NUM_CLASSES, pretrained=False,
                         freeze_backbone=True)
    bb_params = [p for module in m.get_backbone_modules()
                 for p in module.parameters()]
    assert all(not p.requires_grad for p in bb_params)
    head_params = list(m.get_head().parameters())
    assert all(p.requires_grad for p in head_params)


def test_unfreeze_top_n_layers():
    m = ResNetClassifier(num_classes=NUM_CLASSES, pretrained=False,
                         freeze_backbone=True)
    m.unfreeze_top_n_layers(n=2)
    last_two = m.get_backbone_modules()[-2:]
    for module in last_two:
        assert any(p.requires_grad for p in module.parameters())


@pytest.mark.slow
def test_model_overfits_small_batch():
    """Sanity check that the head can fit a 4-sample batch."""
    torch.manual_seed(0)
    m = ResNetClassifier(num_classes=NUM_CLASSES, pretrained=False,
                         freeze_backbone=True)
    optimizer = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=1e-2
    )
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])
    loss_fn = torch.nn.CrossEntropyLoss()
    init_loss = loss_fn(m(x), y).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_fn(m(x), y)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(m(x), y).item()
    assert final_loss < init_loss
