"""ResNet image encoder for video action segmentation.

This module provides the ImageEncoderResNet class, which wraps
ResNet models for use as a frame encoder in video action segmentation.

The encoder outputs spatial feature maps that can be pooled or processed
by downstream modules, matching the interface of ImageEncoderViTMAE.

Note: This encoder initializes with random weights. Use load_pretrained_weights()
to load a checkpoint file (e.g., from a pretrained autoencoder or classifier).

Architecture:
    Input: Image tensor (B, C, H, W)
    1. Conv stem: Initial convolution and pooling
    2. ResNet stages: Four residual stages with downsampling
    3. Output: Spatial features (B, hidden_size, H', W')
    
    Where H' = W' = image_size / 32 (typically 7 for 224/32)

Supported backbones:
    - resnet18, resnet34: 512-dim output features
    - resnet50, resnet101, resnet152: 2048-dim output features

Reference:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    https://arxiv.org/abs/1512.03385
"""

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from typeguard import typechecked


# Mapping of backbone names to output dimensions
RESNET_HIDDEN_SIZES = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}

# Mapping of backbone names to torchvision model constructors and weights
RESNET_MODELS = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
    'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
    'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


class ImageEncoderResNet(nn.Module):
    """Image encoder using ResNet backbone.
    
    This encoder processes individual frames to extract spatial features.
    It uses torchvision's ResNet implementation.
    
    Note: Initializes with random weights. Call load_pretrained_weights()
    with your checkpoint file to load pretrained weights.
    
    The encoder outputs spatial feature maps that preserve spatial structure,
    matching the interface of ImageEncoderViTMAE for interchangeable use.
    
    Attributes:
        backbone: The underlying ResNet model (without final pooling/FC).
        
    Properties:
        hidden_size: Dimension of output features.
        num_channels: Number of input image channels (3 for RGB).
        image_size: Expected input image size (224).
        patch_size: Effective "patch size" (stride, 32 for ResNet).
    
    Example:
        config = {'backbone': 'resnet50'}
        encoder = ImageEncoderResNet(config)
        
        # Load pretrained weights (required for useful features)
        encoder.load_pretrained_weights('path/to/checkpoint.ckpt')
        
        # Encode images
        images = torch.randn(4, 3, 224, 224)
        features = encoder(images)  # (4, 2048, 7, 7)
    """
    
    @typechecked
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ResNet image encoder.
        
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
        
        # Validate backbone name
        if self._backbone_name not in RESNET_MODELS:
            raise ValueError(
                f"Unsupported backbone: {self._backbone_name}. "
                f"Supported: {list(RESNET_MODELS.keys())}"
            )
        
        # Get output dimension for this backbone
        self._hidden_size = RESNET_HIDDEN_SIZES[self._backbone_name]
        
        # Store config for reference
        self._config = config
        
        # Build backbone
        self._build_backbone()
    
    def _build_backbone(self) -> None:
        """Build the ResNet backbone, removing the final pooling and FC layers.
        
        Initializes with random weights. Call load_pretrained_weights() to
        load a checkpoint.
        """
        model_fn, _ = RESNET_MODELS[self._backbone_name]
        
        # Initialize with random weights (no pretrained)
        # Use load_pretrained_weights() to load checkpoint
        full_model = model_fn(weights=None)
        
        # Remove average pooling and FC layers to get spatial features
        # ResNet structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.backbone = nn.Sequential(
            full_model.conv1,      # 7x7 conv, stride 2
            full_model.bn1,
            full_model.relu,
            full_model.maxpool,    # 3x3 pool, stride 2
            full_model.layer1,     # No downsampling
            full_model.layer2,     # Stride 2
            full_model.layer3,     # Stride 2
            full_model.layer4,     # Stride 2
        )
        # Total downsampling: 2 * 2 * 2 * 2 * 2 = 32x
    
    @property
    def hidden_size(self) -> int:
        """Dimension of output features."""
        return self._hidden_size
    
    @property
    def num_channels(self) -> int:
        """Number of input image channels."""
        return 3
    
    @property
    def image_size(self) -> int:
        """Expected input image size (square)."""
        return self._image_size
    
    @property
    def patch_size(self) -> int:
        """Effective stride/patch size for the backbone.
        
        For ResNet, this is the total downsampling factor (32).
        This property exists for API compatibility with ViT encoders.
        """
        return 32
    
    @property
    def backbone_name(self) -> str:
        """Name of the ResNet variant being used."""
        return self._backbone_name
    
    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return 'resnet'
    
    @typechecked
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet encoder.
        
        Processes input images through the ResNet backbone and
        returns spatial feature maps.
        
        Args:
            x: Input images, shape (B, C, H, W).
                Expected to be normalized with ImageNet statistics:
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Returns:
            Spatial features, shape (B, hidden_size, H', W').
            For 224x224 input: H' = W' = 7
        
        Raises:
            ValueError: If input channels don't match expected channels.
        """
        B, C, H, W = x.shape
        
        # Validate input channels
        if C != self.num_channels:
            raise ValueError(
                f"Input has {C} channels but model expects {self.num_channels}"
            )
        
        # Forward through ResNet backbone
        # Output shape: (B, hidden_size, H/32, W/32)
        spatial_features = self.backbone(x)
        
        return spatial_features
    
    @typechecked
    def load_pretrained_weights(
        self, 
        checkpoint_path: str, 
        strict: bool = False,
    ) -> None:
        """Load pretrained ResNet encoder weights with shape filtering.
        
        This method loads weights from a checkpoint file, handling different
        checkpoint formats (Lightning, plain PyTorch) and filtering for
        compatible weights.
        
        Args:
            checkpoint_path: Path to checkpoint file (.ckpt or .pt).
            strict: If True, raise error on missing/extra keys.
                If False (default), load only matching weights.
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        current_state_dict = self.backbone.state_dict()
        
        # Try various key prefix patterns
        prefixes_to_try = [
            'encoder.backbone.',  # From VideoSegmenter checkpoint
            'backbone.',          # Direct backbone save
            'resnet.',            # Named as resnet
            'encoder.',           # Generic encoder prefix
            '',                   # No prefix
        ]
        
        filtered_state_dict = {}
        
        for ckpt_key, value in state_dict.items():
            # Skip FC layer weights (we don't use them)
            if 'fc.' in ckpt_key:
                continue
            
            # Try each prefix
            for prefix in prefixes_to_try:
                if prefix and ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                elif not prefix:
                    model_key = ckpt_key
                else:
                    continue
                
                # Check if key exists and shape matches
                if model_key in current_state_dict:
                    if current_state_dict[model_key].shape == value.shape:
                        if model_key not in filtered_state_dict:
                            filtered_state_dict[model_key] = value
                        break
        
        if filtered_state_dict:
            self.backbone.load_state_dict(filtered_state_dict, strict=strict)
            print(f"Loaded {len(filtered_state_dict)} weights from checkpoint")
        else:
            print("Warning: No matching weights found in checkpoint")
    
    def get_last_layer_params(self):
        """Get parameters of the last residual stage for fine-tuning.
        
        Returns:
            Iterator over parameters of layer4 (last residual stage).
        """
        # backbone[7] is layer4 in our Sequential
        for param in self.backbone[7].parameters():
            yield param
