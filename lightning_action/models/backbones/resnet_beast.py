"""Beast ResNet backbone for video action segmentation.

This module provides the ResNetBeastBackbone class, which wraps the
custom ResNet implementation from the beast package for use as a frame
backbone in video action segmentation.

This backbone is compatible with checkpoints trained using the beast
ResnetAutoEncoder. It uses the backbone portion only.

Architecture:
    Input: Image tensor (B, C, H, W)
    1. conv1: 7x7 conv, stride 2 (224 -> 112)
    2. conv2: residual block with pool (112 -> 56)
    3. conv3: residual block with stride (56 -> 28)
    4. conv4: residual block with stride (28 -> 14)
    5. conv5: residual block with stride (14 -> 7)
    Output: Spatial features (B, hidden_dim, 7, 7)

Hidden dimensions:
    - resnet18/34 (no bottleneck): 512
    - resnet50/101/152 (bottleneck): 2048
"""

from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional

import torch
import torch.nn as nn
from typeguard import typechecked


@typechecked
def get_configs(arch: str = 'resnet50') -> tuple:
    """Get number and type of layers for resnet models.
    
    Args:
        arch: ResNet architecture name.
    
    Returns:
        Tuple of (layer_configs, use_bottleneck).
    
    Raises:
        ValueError: If architecture is not supported.
    """
    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError(f'{arch} is not a valid ResNet architecture')


# Hidden sizes for each architecture
BEAST_RESNET_HIDDEN_SIZES = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}


class ResNetBeastBackbone(nn.Module):
    """Image backbone using beast's custom ResNet implementation.
    
    This backbone is compatible with checkpoints trained using beast's
    ResnetAutoEncoder. It extracts only the backbone portion.
    
    Attributes:
        backbone: The ResNetBeast module.
    
    Properties:
        hidden_size: Output feature dimension.
        num_channels: Number of input channels (3 for RGB).
        image_size: Expected input image size (224).
        patch_size: Effective patch size (32, due to total stride).
        backbone_type: Returns 'resnet-beast'.
    
    Example:
        config = {'backbone': 'resnet50'}
        backbone = ResNetBeastBackbone(config)
        
        images = torch.randn(4, 3, 224, 224)
        features = backbone(images)  # (4, 2048, 7, 7)
    """
    
    @typechecked
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the beast ResNet backbone.
        
        Args:
            config: Configuration dictionary with optional keys:
                - backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50',
                    'resnet101', 'resnet152'). Default: 'resnet50'.
                - image_size: Expected input image size. Default: 224.
        
        Raises:
            ValueError: If backbone name is not supported.
        """
        super().__init__()
        
        config = config or {}
        
        # Parse configuration
        self._backbone_name = config.get('backbone', 'resnet50')
        self._image_size = config.get('image_size', 224)
        
        # Validate and get architecture config
        if self._backbone_name not in BEAST_RESNET_HIDDEN_SIZES:
            raise ValueError(
                f"Unsupported backbone: {self._backbone_name}. "
                f"Supported: {list(BEAST_RESNET_HIDDEN_SIZES.keys())}"
            )
        
        self._hidden_size = BEAST_RESNET_HIDDEN_SIZES[self._backbone_name]
        self._config = config
        
        # Build backbone
        resnet_config, bottleneck = get_configs(self._backbone_name)
        self.backbone = ResNetBeast(configs=resnet_config, bottleneck=bottleneck)
    
    @property
    def hidden_size(self) -> int:
        """Output feature dimension."""
        return self._hidden_size
    
    @property
    def num_channels(self) -> int:
        """Number of input image channels."""
        return 3
    
    @property
    def image_size(self) -> int:
        """Expected input image size."""
        return self._image_size
    
    @property
    def patch_size(self) -> int:
        """Effective patch size (total stride of the network)."""
        return 32  # 224 -> 7 means stride of 32
    
    @property
    def backbone_type(self) -> str:
        """Return backbone type identifier."""
        return 'resnet-beast'
    
    @typechecked
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.
        
        Args:
            x: Input images, shape (B, C, H, W).
        
        Returns:
            Spatial features, shape (B, hidden_size, H', W').
            For 224x224 input: (B, hidden_size, 7, 7).
        """
        return self.backbone(x)
    
    @typechecked
    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> None:
        """Load pretrained weights from a beast checkpoint.
        
        Handles beast's ResnetAutoEncoder checkpoint format, extracting
        only the encoder weights.
        
        Args:
            checkpoint_path: Path to checkpoint file (.ckpt or .pt).
            strict: If True, raise error on missing/extra keys.
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Extract encoder weights with various possible prefixes
        encoder_state_dict = {}
        current_state_dict = self.backbone.state_dict()
        
        prefixes_to_try = [
            'encoder.',           # Direct encoder save
            'model.encoder.',     # Wrapped in model
            '',                   # No prefix
        ]
        
        for ckpt_key, value in state_dict.items():
            # Skip decoder and latent mapping weights
            if 'decoder' in ckpt_key or 'latent' in ckpt_key.lower():
                continue
            
            for prefix in prefixes_to_try:
                if prefix and ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                elif not prefix:
                    model_key = ckpt_key
                else:
                    continue
                
                if model_key in current_state_dict:
                    if current_state_dict[model_key].shape == value.shape:
                        encoder_state_dict[model_key] = value
                        break
        
        if encoder_state_dict:
            self.backbone.load_state_dict(encoder_state_dict, strict=strict)
            print(f"Loaded {len(encoder_state_dict)} weights from checkpoint")
        else:
            print("Warning: No matching weights found in checkpoint")
    
    def get_last_layer_params(self) -> Iterator[nn.Parameter]:
        """Get parameters of the last encoder block for fine-tuning.
        
        Returns:
            Iterator over parameters of conv5 (last residual block).
        """
        return self.backbone.conv5.parameters()


# =============================================================================
# Beast ResNet Backbone Components (from beast/models/resnet_ae.py)
# =============================================================================

class ResNetBeast(nn.Module):
    """ResNet backbone from beast package."""

    def __init__(self, configs: list, bottleneck: bool = False) -> None:
        super().__init__()

        if len(configs) != 4:
            raise ValueError('Only 4 layers can be configured')

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:
            self.conv2 = BottleneckBlock(
                in_channels=64, hidden_channels=64, up_channels=256, layers=configs[0],
                downsample_method='pool',
            )
            self.conv3 = BottleneckBlock(
                in_channels=256, hidden_channels=128, up_channels=512, layers=configs[1],
                downsample_method='conv',
            )
            self.conv4 = BottleneckBlock(
                in_channels=512, hidden_channels=256, up_channels=1024, layers=configs[2],
                downsample_method='conv',
            )
            self.conv5 = BottleneckBlock(
                in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3],
                downsample_method='conv',
            )
        else:
            self.conv2 = ResidualBlock(
                in_channels=64, hidden_channels=64, layers=configs[0], downsample_method='pool',
            )
            self.conv3 = ResidualBlock(
                in_channels=64, hidden_channels=128, layers=configs[1], downsample_method='conv',
            )
            self.conv4 = ResidualBlock(
                in_channels=128, hidden_channels=256, layers=configs[2], downsample_method='conv',
            )
            self.conv5 = ResidualBlock(
                in_channels=256, hidden_channels=512, layers=configs[3], downsample_method='conv',
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for non-bottleneck ResNets (18, 34)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:
        super().__init__()

        if downsample_method == 'conv':
            for i in range(layers):
                if i == 0:
                    layer = ResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=True,
                    )
                else:
                    layer = ResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )
                self.add_module(f'{i} EncoderLayer', layer)

        elif downsample_method == 'pool':
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):
                if i == 0:
                    layer = ResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )
                else:
                    layer = ResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )
                self.add_module(f'{i + 1} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block for deeper ResNets (50, 101, 152)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        layers: int,
        downsample_method: Literal['conv', 'pool'] = 'conv',
    ) -> None:
        super().__init__()

        if downsample_method == 'conv':
            for i in range(layers):
                if i == 0:
                    layer = BottleneckLayer(
                        in_channels=in_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=True,
                    )
                else:
                    layer = BottleneckLayer(
                        in_channels=up_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )
                self.add_module(f'{i} EncoderLayer', layer)

        elif downsample_method == 'pool':
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):
                if i == 0:
                    layer = BottleneckLayer(
                        in_channels=in_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )
                else:
                    layer = BottleneckLayer(
                        in_channels=up_channels, hidden_channels=hidden_channels,
                        up_channels=up_channels, downsample=False,
                    )
                self.add_module(f'{i + 1} EncoderLayer', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, layer in self.named_children():
            x = layer(x)
        return x


class ResidualLayer(nn.Module):
    """Single residual layer for non-bottleneck ResNets."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        downsample: bool,
    ) -> None:
        super().__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                    stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                    stride=1, padding=1, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.relu(x)
        return x


class BottleneckLayer(nn.Module):
    """Single bottleneck layer for deeper ResNets."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        up_channels: int,
        downsample: bool,
    ) -> None:
        super().__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=up_channels, kernel_size=1,
                stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=up_channels, kernel_size=1,
                    stride=2, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif in_channels != up_channels:
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=up_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)
        x = x + identity
        x = self.relu(x)
        return x
