"""Backbone architectures for action segmentation models."""

from lightning_action.models.backbones.resnet import ResNetBackbone
from lightning_action.models.backbones.resnet_beast import ResNetBeastBackbone
from lightning_action.models.backbones.vitmae import ViTMAEBackbone

__all__ = [
    'ResNetBackbone',
    'ResNetBeastBackbone',
    'ViTMAEBackbone',
]
