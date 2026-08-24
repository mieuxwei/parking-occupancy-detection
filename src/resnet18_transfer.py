"""ResNet18 transfer-learning model construction and freezing helpers."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(pretrained: bool = True, dropout: float = 0.20) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
