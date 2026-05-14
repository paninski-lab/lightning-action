"""ViT-MAE image encoder for video action segmentation.

This module provides the ViTMAEBackbone class, which wraps a Vision
Transformer with Masked Autoencoding (ViT-MAE) for use as a backbone
in video action segmentation.

Architecture:
    Input: Image tensor (B, C, H, W)
    1. Patch embedding: (B, C, H, W) -> (B, num_patches, hidden_dim)
    2. Transformer encoder: Self-attention over patches
    3. Output: Spatial features (B, hidden_dim, H', W')

    Where H' = W' = image_size / patch_size (typically 14 for 224/16)

Reference:
    He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
    https://arxiv.org/abs/2111.06377
"""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import ViTMAEModel


class ViTMAEBackbone(nn.Module):
    """Image encoder using ViT-MAE with optional pretrained weights.

    This backbone processes individual frames to extract spatial features.
    It uses the HuggingFace transformers implementation of ViT-MAE.

    The backbone outputs spatial feature maps that preserve the patch grid
    structure, which can be useful for spatial attention or pooling.

    Attributes:
        vit_mae: The underlying ViT-MAE model from HuggingFace.

    Properties:
        hidden_size: Dimension of transformer hidden states.
        num_channels: Number of input image channels (typically 3 for RGB).
        image_size: Expected input image size (square).
        patch_size: Size of image patches for the transformer.

    Example:
        # With config
        config = {'backbone': 'vitmae'}
        backbone = ViTMAEBackbone(config)

        # Load pretrained weights
        backbone.load_pretrained_weights('path/to/checkpoint.ckpt')

        # Encode images
        images = torch.randn(4, 3, 224, 224)
        features = backbone(images)  # (4, 768, 14, 14)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the ViT-MAE backbone.

        Args:
            config: Configuration dictionary with optional keys:
                - model_name: HuggingFace model identifier
                    (default: 'facebook/vit-mae-base')
                - mask_ratio: Masking ratio, should be 0.0 for inference
                    (default: 0.0)
        """
        super().__init__()

        config = config or {}

        # Get model configuration
        model_name = config.get('model_name', 'facebook/vit-mae-base')
        mask_ratio = config.get('mask_ratio', 0.0)

        # Initialize ViT-MAE from HuggingFace
        self.vit_mae = ViTMAEModel.from_pretrained(
            model_name,
            mask_ratio=mask_ratio,
        )

        # Store config for reference
        self._config = config

    @property
    def hidden_size(self) -> int:
        """Dimension of transformer hidden states."""
        return self.vit_mae.config.hidden_size

    @property
    def num_channels(self) -> int:
        """Number of input image channels."""
        return self.vit_mae.config.num_channels

    @property
    def image_size(self) -> int:
        """Expected input image size (square)."""
        return self.vit_mae.config.image_size

    @property
    def patch_size(self) -> int:
        """Size of image patches for the transformer."""
        return self.vit_mae.config.patch_size

    @property
    def backbone_type(self) -> str:
        """Return backbone type identifier."""
        return 'vitmae'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT-MAE backbone.

        Processes input images through the transformer backbone and
        reshapes the output to a spatial feature map.

        Args:
            x: Input images, shape (B, C, H, W).
                Expected to be normalized with ImageNet statistics.

        Returns:
            Spatial features, shape (B, hidden_size, H', W').
            H' = W' = image_size // patch_size

        Raises:
            ValueError: If input channels don't match expected channels.
        """
        B, C, H, W = x.shape

        # Validate input channels
        if C != self.num_channels:
            raise ValueError(
                f"Input has {C} channels but model expects {self.num_channels}"
            )

        # Forward through ViT-MAE
        # noise=None ensures no random masking is applied
        outputs = self.vit_mae(
            pixel_values=x,
            noise=None,
            output_hidden_states=False,
            return_dict=True,
        )

        # Extract hidden states: (B, 1 + num_patches, hidden_dim)
        # First token is [CLS], rest are patch tokens
        hidden_states = outputs.last_hidden_state

        # Remove [CLS] token, keep only patch embeddings
        patch_embeddings = hidden_states[:, 1:, :]  # (B, num_patches, hidden_dim)

        # Reshape to spatial grid
        num_patches = patch_embeddings.shape[1]
        H_out = W_out = int(math.sqrt(num_patches))

        spatial_features = patch_embeddings.reshape(B, H_out, W_out, self.hidden_size)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, hidden_dim, H', W')

        return spatial_features

    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> None:
        """Load pretrained ViT-MAE encoder weights with shape filtering.

        This method loads weights from a checkpoint file, handling the case
        where the checkpoint may contain additional keys (like decoder weights)
        or have different layer configurations.

        Only weights that:
        1. Belong to the encoder (not decoder)
        2. Have matching shapes with current model

        will be loaded.

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

        current_state_dict = self.vit_mae.state_dict()

        # Filter and rename keys for compatibility
        encoder_state_dict = {}

        # Try multiple prefix patterns
        prefixes_to_try = [
            'vit_mae.vit.',      # From Lightning checkpoint with this encoder
            'encoder.vit_mae.',  # From VideoSegmenter checkpoint
            'vit.',              # Direct ViT save
            '',                  # No prefix
        ]

        for ckpt_key, value in state_dict.items():
            # Skip decoder weights and mask tokens
            if 'decoder' in ckpt_key.lower() or 'mask_token' in ckpt_key:
                continue

            # Try each prefix
            for prefix in prefixes_to_try:
                if prefix and ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                elif not prefix:
                    model_key = ckpt_key
                else:
                    continue

                # Only load if key exists and shape matches
                if model_key in current_state_dict:
                    if current_state_dict[model_key].shape == value.shape:
                        encoder_state_dict[model_key] = value
                        break

        # Load filtered weights
        if encoder_state_dict:
            self.vit_mae.load_state_dict(encoder_state_dict, strict=strict)
            print(f"Loaded {len(encoder_state_dict)} weights from checkpoint")
        else:
            print("Warning: No matching weights found in checkpoint")

    def get_last_layer_params(self):
        """Get parameters of the last transformer layer for fine-tuning.

        Returns:
            Iterator over parameters of the last encoder layer and layernorm.
        """
        for param in self.vit_mae.encoder.layer[-1].parameters():
            yield param
        for param in self.vit_mae.layernorm.parameters():
            yield param
