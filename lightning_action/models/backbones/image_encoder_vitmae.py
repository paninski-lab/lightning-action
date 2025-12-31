"""ViT-MAE image encoder for video action segmentation.

This module provides the ImageEncoderViTMAE class, which wraps a Vision
Transformer with Masked Autoencoding (ViT-MAE) for use as a frame encoder
in video action segmentation.

ViT-MAE was pretrained on large image datasets using a masked autoencoding
objective, learning rich visual representations. We use only the encoder
portion (discarding the decoder) to extract features from video frames.

Architecture:
    Input: Image tensor (B, C, H, W)
    1. Patch embedding: (B, C, H, W) -> (B, num_patches, hidden_dim)
    2. Transformer encoder: Self-attention over patches
    3. Output: Spatial features (B, hidden_dim, H', W')
    
    Where H' = W' = image_size / patch_size (typically 14 for 224/16)

The output is a spatial feature map that can be pooled or processed
further by downstream modules.

Reference:
    He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
    https://arxiv.org/abs/2111.06377
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTMAEModel


class ImageEncoderViTMAE(nn.Module):
    """Image encoder using ViT-MAE with optional pretrained weights.
    
    This encoder processes individual frames to extract spatial features.
    It uses the HuggingFace transformers implementation of ViT-MAE.
    
    The encoder outputs spatial feature maps that preserve the patch grid
    structure, which can be useful for spatial attention or pooling.
    
    Attributes:
        vit_mae: The underlying ViT-MAE model from HuggingFace.
        
    Properties:
        hidden_size: Dimension of transformer hidden states.
        num_channels: Number of input image channels (typically 3 for RGB).
        image_size: Expected input image size (square).
        patch_size: Size of image patches for the transformer.
    
    Example:
        encoder = ImageEncoderViTMAE()
        
        # Load pretrained weights
        encoder.load_pretrained_weights('path/to/checkpoint.ckpt')
        
        # Encode images
        images = torch.randn(4, 3, 224, 224)
        features = encoder(images)  # (4, 768, 14, 14)
    """
    
    def __init__(self, config: dict = None):
        """Initialize the ViT-MAE image encoder.
        
        Args:
            config: Optional configuration dictionary. Currently unused as
                model configuration is determined by the pretrained weights.
                Kept for API compatibility.
        """
        super().__init__()
        
        # Initialize ViT-MAE from HuggingFace with Facebook's pretrained weights
        # We use mask_ratio=0 since we want full image encoding (no masking)
        self.vit_mae = ViTMAEModel.from_pretrained(
            "facebook/vit-mae-base",
            mask_ratio=0.0,  # No masking during feature extraction
        )
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT-MAE encoder.
        
        Processes input images through the transformer encoder and
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
        state_dict = checkpoint.get('state_dict', checkpoint)
        current_state_dict = self.vit_mae.state_dict()
        
        # Filter and rename keys for compatibility
        encoder_state_dict = {}
        PREFIX_TO_STRIP = 'vit_mae.vit.'
        
        for ckpt_key, value in state_dict.items():
            # Skip decoder weights and mask tokens
            if ckpt_key.startswith('vit_mae.decoder.') or 'mask_token' in ckpt_key:
                continue
            
            # Strip prefix if present
            if ckpt_key.startswith(PREFIX_TO_STRIP):
                model_key = ckpt_key[len(PREFIX_TO_STRIP):]
                
                # Only load if key exists and shape matches
                if model_key in current_state_dict:
                    if current_state_dict[model_key].shape == value.shape:
                        encoder_state_dict[model_key] = value
        
        # Load filtered weights
        self.vit_mae.load_state_dict(encoder_state_dict, strict=strict)
    
    def eval(self) -> 'ImageEncoderViTMAE':
        """Set encoder to evaluation mode.
        
        Disables dropout and other training-specific behaviors.
        
        Returns:
            Self for method chaining.
        """
        super().eval()
        self.vit_mae.eval()
        return self
    
    def train(self, mode: bool = True) -> 'ImageEncoderViTMAE':
        """Set encoder to training or evaluation mode.
        
        Args:
            mode: If True, set to training mode. If False, evaluation mode.
        
        Returns:
            Self for method chaining.
        """
        super().train(mode)
        self.vit_mae.train(mode)
        return self
