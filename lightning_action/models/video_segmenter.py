"""Video segmentation model with image encoder backbone and temporal modeling head.

This module implements a video action segmentation model that:
1. Encodes individual frames using a ViT-MAE (Vision Transformer with Masked Autoencoding)
2. Pools spatial features from each frame into a single vector
3. Models temporal dynamics using a head network (TCN, MLP, or RNN)
4. Classifies each frame into an action class

VideoBaseModel inherits from BaseModel (models/segmenter.py) and overrides
only the methods that need video-specific behavior.
"""

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from lightning_action.models.backbones import (
    ResNetBackbone,
    ResNetBeastBackbone,
    ViTMAEBackbone,
)
from lightning_action.models.heads import RNN, DilatedTCN, TemporalMLP
from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
from lightning_action.models.segmenter import BaseModel


class VideoBaseModel(BaseModel):
    """Base model for video action segmentation.

    Inherits common functionality from BaseModel and adds video-specific
    behavior for handling:
    - Tuple-based batches (frames, labels, metadata) from DALI
    - Boundary-aware prediction slicing for video chunks
    - DDP coordination for skipping all-ignored batches

    Subclasses must implement:
    - _build_model(): Construct the model architecture
    - forward(): Define the forward pass
    """

    def _get_inputs_and_targets(
        self,
        batch: tuple | dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict] | None]:
        """Extract inputs and targets from batch.

        Handles both tuple format (from DALI) and dict format (from standard loaders).

        Args:
            batch: Either tuple (frames, labels, metadata) or dict with 'input'/'labels'.

        Returns:
            Tuple of (inputs, targets, metadata).
        """
        if isinstance(batch, dict):
            return batch['input'], batch.get('labels'), None

        if len(batch) == 3:
            frames, labels, metadata = batch
        elif len(batch) == 2:
            frames, labels = batch
            metadata = None
        else:
            frames = batch[0]
            labels = None
            metadata = None

        return frames, labels, metadata

    @typechecked
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[dict]] | dict,
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Execute one training step.

        Handles the special case where all labels in a batch are ignore_index
        by skipping the batch (returns None). For DDP, coordinates across
        all GPUs to skip only if ALL GPUs have all-ignore batches.

        Args:
            batch: Tuple of (frames, labels, metadata) or dict.
            batch_idx: Index of this batch.

        Returns:
            Loss tensor, or None if batch should be skipped.
        """
        frames, labels, metadata = self._get_inputs_and_targets(batch)

        # Removes the start and end chunks during training to prevent context corruption.
        if metadata is not None:
            is_partial = any(
                m.get('is_start', False) or m.get('is_end', False)
                for m in metadata
            )
            if is_partial:
                return None

        if self.sequence_pad and self.sequence_pad > 0:
            trimmed_labels = labels[:, self.sequence_pad:-self.sequence_pad]
        else:
            trimmed_labels = labels

        all_ignored = torch.all(trimmed_labels == self.ignore_index)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            all_ignored_tensor = all_ignored.float().unsqueeze(0)
            all_ignored_list = [torch.zeros(1, device=self.device) for _ in range(world_size)]
            torch.distributed.all_gather(all_ignored_list, all_ignored_tensor)
            all_ignored_gathered = torch.cat(all_ignored_list)

            if torch.all(all_ignored_gathered > 0.5):
                return None
        elif all_ignored:
            return None

        outputs = self.forward(frames)
        outputs_no_pad = self._remove_padding(outputs)
        labels_no_pad = self._remove_padding(labels)

        loss, metrics = self.compute_loss(outputs_no_pad, labels_no_pad, stage='train')

        if metrics:
            self.log_dict(
                metrics,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=frames.shape[0],
            )

        return loss

    @typechecked
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[dict]] | dict,
        batch_idx: int,
    ) -> None:
        """Execute one validation step.

        Args:
            batch: Tuple of (frames, labels, metadata) or dict.
            batch_idx: Index of this batch.
        """
        frames, labels, _ = self._get_inputs_and_targets(batch)

        outputs = self.forward(frames)
        outputs_no_pad = self._remove_padding(outputs)
        labels_no_pad = self._remove_padding(labels)

        loss, metrics = self.compute_loss(outputs_no_pad, labels_no_pad, stage='val')

        if metrics:
            self.log_dict(
                metrics,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=frames.shape[0],
            )

    @typechecked
    def predict_step(
        self,
        batch: tuple[torch.Tensor, list[int], list[dict]] | dict,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> list[torch.Tensor]:
        """Execute one prediction step with boundary-aware output slicing.

        The extended sequence from DALI provides context on both sides.
        For boundary chunks (first/last in a video), we adjust slicing to
        include predictions for frames that would normally be cut off.

        Args:
            batch: Tuple of (frames, actual_lengths, metadata) or dict.
            batch_idx: Index of this batch.
            dataloader_idx: Index of dataloader (for multiple dataloaders).

        Returns:
            List of probability tensors, one per sample in batch.
        """
        frames, _, metadata = self._get_inputs_and_targets(batch)

        is_start = None
        is_end = None
        if metadata is not None:
            is_start = [m.get('is_start', False) for m in metadata]
            is_end = [m.get('is_end', False) for m in metadata]

        outputs = self.forward(frames)
        probabilities = outputs['probabilities']

        result = []
        pad = self.sequence_pad or 0

        for i in range(probabilities.shape[0]):
            sample_probs = probabilities[i]
            sample_is_start = is_start[i] if is_start else False
            sample_is_end = is_end[i] if is_end else False

            if pad > 0:
                if sample_is_start and sample_is_end:
                    valid_probs = sample_probs[pad:-pad]
                elif sample_is_start:
                    valid_probs = sample_probs[:-pad]
                elif sample_is_end:
                    valid_probs = sample_probs[pad:]
                else:
                    valid_probs = sample_probs[pad:-pad]
            else:
                valid_probs = sample_probs

            result.append(valid_probs)

        return result

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch. Clears GPU cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_optimizer_params(self):
        """Get parameters to optimize.

        Override in subclasses for custom parameter groups.
        Default returns self.parameters().

        Returns:
            Parameters or list of parameter group dicts.
        """
        return self.parameters()


class VideoSegmenter(VideoBaseModel):
    """Video action segmentation model with swappable backbone.

    This model processes video frames through:
    1. Configurable backbone: Extract spatial features from each frame
       - ViT-MAE: Patch-level transformer features (hidden_size=768)
       - ResNet: Convolutional feature maps (hidden_size=512-2048)
    2. Attention pooling: Aggregate spatial features per frame
    3. Feature augmentation: Concatenate positions with velocities
    4. Temporal head: Model dynamics (TCN, MLP, or RNN)
    5. Classifier: Per-frame action prediction

    Attributes:
        backbone: Image encoder (ViT-MAE or ResNet).
        pooling: MultiheadAttentionPooling for spatial pooling.
        head: Temporal sequence model.
        classifier: Linear classification head.
    """

    # Hidden sizes for each backbone type (used for input_size auto-computation)
    BACKBONE_HIDDEN_SIZES = {
        'vitmae': 768,
        'vit-mae': 768,
        'vit': 768,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        # Beast variants (same hidden sizes, different architecture)
        'resnet18-beast': 512,
        'resnet34-beast': 512,
        'resnet50-beast': 2048,
        'resnet101-beast': 2048,
        'resnet152-beast': 2048,
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize VideoSegmenter with auto-computed input_size.

        Args:
            config: Configuration dictionary.
        """
        # Auto-compute input_size if not provided, before BaseModel.__init__ runs
        model_config = config.get('model', {})
        if model_config.get('input_size') is None:
            backbone_name = model_config.get('backbone', 'vitmae').lower()
            hidden_size = self.BACKBONE_HIDDEN_SIZES.get(backbone_name, 768)
            # input_size = 2 * hidden_size (position + velocity features)
            config['model']['input_size'] = hidden_size * 2

        super().__init__(config)

    def _build_model(self) -> None:
        """Build the complete video segmentation model."""
        # Load backbone config from file if provided
        backbone_file_config = {}
        backbone_config_path = self.model_config.get('backbone_config_path')
        if backbone_config_path and os.path.exists(backbone_config_path):
            import yaml
            with open(backbone_config_path) as f:
                backbone_file_config = yaml.safe_load(f)

        # Extract model_params from backbone config (mirrors vit.yaml / resnet_ae.yaml structure)
        backbone_model_params = backbone_file_config.get('model', {}).get('model_params', {})

        # Get backbone name: main config 'backbone' overrides backbone_config_path
        backbone_name = self.model_config.get('backbone')
        if backbone_name is None:
            # Fall back to model_class or model_params.backbone from backbone config
            backbone_name = backbone_file_config.get('model', {}).get('model_class')
            if backbone_name == 'resnet':
                backbone_name = backbone_model_params.get('backbone', 'resnet50')
            elif backbone_name is None:
                backbone_name = 'vitmae'
        backbone_name = backbone_name.lower()

        # Build backbone based on name
        if backbone_name in ['vitmae', 'vit-mae', 'vit']:
            self.backbone_type = 'vitmae'
            vitmae_config = {
                # model_name at top level of backbone config
                'model_name': backbone_file_config.get(
                    'model_name',
                    self.model_config.get('backbone_model_name', 'facebook/vit-mae-base'),
                ),
                # mask_ratio in model.model_params
                'mask_ratio': backbone_model_params.get(
                    'mask_ratio',
                    self.model_config.get('backbone_mask_ratio', 0.0),
                ),
            }
            self.backbone = ViTMAEBackbone(vitmae_config)

        elif backbone_name.endswith('-beast'):
            # Beast ResNet variant (custom implementation from beast package)
            self.backbone_type = 'resnet-beast'
            # e.g., 'resnet50-beast' -> 'resnet50'
            backbone_name = backbone_name.replace('-beast', '')
            resnet_config = {
                'backbone': backbone_name,
                'image_size': backbone_model_params.get(
                    'image_size',
                    self.model_config.get('backbone_image_size', 224),
                ),
            }
            self.backbone = ResNetBeastBackbone(resnet_config)

        elif backbone_name.startswith('resnet'):
            # Standard torchvision ResNet
            self.backbone_type = 'resnet'
            resnet_config = {
                'backbone': backbone_name,
                'image_size': backbone_model_params.get(
                    'image_size',
                    self.model_config.get('backbone_image_size', 224),
                ),
            }
            self.backbone = ResNetBackbone(resnet_config)

        else:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Supported: 'vitmae', "
                f"'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', "
                f"'resnet18-beast', 'resnet34-beast', 'resnet50-beast', 'resnet101-beast', "
                f"'resnet152-beast'"
            )

        # Load backbone checkpoint if specified
        backbone_ckpt = self.model_config.get('backbone_checkpoint')
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            self.backbone.load_pretrained_weights(backbone_ckpt)

        # Get backbone hidden size
        self.embed_dim = self.backbone.hidden_size

        # Validate input_size matches backbone (it was auto-computed in __init__ if not set)
        expected_input_size = self.embed_dim * 2
        if self.input_size != expected_input_size:
            import warnings
            warnings.warn(
                f"Config input_size ({self.input_size}) differs from expected "
                f"({expected_input_size} = 2 × {self.embed_dim}). "
                f"Using config value."
            )

        # Configure backbone freezing
        self.freeze_backbone = self.model_config.get('freeze_backbone', True)
        self._configure_backbone_freezing()

        # Spatial pooling using learned queries
        self.pooling = MultiheadAttentionPooling(
            embed_dim=self.embed_dim,
            num_heads=8,
            num_seeds=1,
            dropout=0.0,
            use_ffn=True,
            layer_norm=False,
        )

        # Build temporal head
        self.head = self._build_head()

        # Classification head
        head_output_size = self._get_head_output_size()
        self.classifier = nn.Linear(head_output_size, self.output_size)

        # Initialize weights
        self._initialize_weights()

    def _configure_backbone_freezing(self) -> None:
        """Configure backbone parameter freezing based on settings."""
        if self.freeze_backbone:
            # Full freeze: no backbone gradients
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Partial fine-tuning: freeze all except last layer
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze last layer using backbone's helper method
            for param in self.backbone.get_last_layer_params():
                param.requires_grad = True

    def _build_head(self) -> nn.Module:
        """Construct temporal head for sequence modeling.

        Returns:
            Configured temporal head module.

        Raises:
            ValueError: If head type is not supported.
        """
        head_type = self.model_config.get('head', 'dtcn')

        if head_type.lower() == 'temporalmlp':
            return TemporalMLP(
                input_size=self.input_size,
                num_hid_units=self.model_config['num_hid_units'],
                num_layers=self.model_config['num_layers'],
                num_lags=self.model_config.get('num_lags', 1),
                activation=self.model_config.get('activation', 'lrelu'),
                dropout_rate=self.model_config.get('dropout_rate', 0.0),
                seed=self.model_config.get('seed', 42),
            )
        elif head_type.lower() == 'rnn':
            return RNN(
                input_size=self.input_size,
                num_hid_units=self.model_config['num_hid_units'],
                num_layers=self.model_config['num_layers'],
                rnn_type=self.model_config.get('rnn_type', 'lstm'),
                bidirectional=self.model_config.get('bidirectional', False),
                dropout_rate=self.model_config.get('dropout_rate', 0.0),
                seed=self.model_config.get('seed', 42),
            )
        elif head_type.lower() in ['dtcn', 'dilatedtcn']:
            return DilatedTCN(
                input_size=self.input_size,
                num_hid_units=self.model_config['num_hid_units'],
                num_layers=self.model_config['num_layers'],
                num_lags=self.model_config.get('num_lags', 1),
                activation=self.model_config.get('activation', 'lrelu'),
                dropout_rate=self.model_config.get('dropout_rate', 0.1),
                seed=self.model_config.get('seed', 42),
            )
        else:
            raise ValueError(f'Unsupported head type: {head_type}')

    def _get_head_output_size(self) -> int:
        """Get the output feature dimension of the head."""
        return self.model_config['num_hid_units']

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.pooling, self.head, self.classifier]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def _get_optimizer_params(self):
        """Get parameters to optimize with specific groups.

        Returns parameters from pooling, head, and classifier.
        Backbone parameters are handled separately based on freeze settings.

        Returns:
            List of parameter group dicts.
        """
        param_groups = [
            {'params': self.pooling.parameters()},
            {'params': self.head.parameters()},
            {'params': self.classifier.parameters()},
        ]

        # Add backbone parameters if not fully frozen
        if not self.freeze_backbone:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if backbone_params:
                # Use lower learning rate for backbone fine-tuning
                backbone_lr = self.config.get('optimizer', {}).get('backbone_lr')
                if backbone_lr is None:
                    base_lr = self.config.get('optimizer', {}).get('lr', 1e-3)
                    backbone_lr = base_lr * 0.1

                param_groups.append({
                    'params': backbone_params,
                    'lr': backbone_lr,
                })

        return param_groups

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence channels height width'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through video segmentation model.

        Processing steps:
        1. Reshape for batch encoding: (B, T, C, H, W) -> (B*T, C, H, W)
        2. Backbone: Extract spatial features (works for both ViT and ResNet)
        3. Spatial pooling: Aggregate to frame-level features
        4. Feature augmentation: Add temporal differences (velocity)
        5. Temporal head: Model dynamics
        6. Classification: Per-frame action prediction

        Args:
            x: Input video frames, shape (batch, time, channels, height, width).

        Returns:
            Dict containing:
            - 'logits': Raw class scores, shape (B, T, num_classes)
            - 'probabilities': Softmax probabilities, shape (B, T, num_classes)
            - 'features': Frame features before head, shape (B, T, 2*embed_dim)
        """
        b, s, c, h, w = x.shape

        # Flatten batch and time for frame-wise encoding
        x = x.view(b * s, c, h, w)

        # Encode frames with selected backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            if self.freeze_backbone:
                self.backbone.eval()
            spatial_features = self.backbone(x)  # (B*T, hidden_dim, H', W')

        # Reshape for attention pooling: (B*T, H'*W', hidden_dim)
        bs, feat_d, feat_h, feat_w = spatial_features.shape
        num_patches = feat_h * feat_w
        patch_features = spatial_features.reshape(bs, feat_d, num_patches)
        patch_features = patch_features.transpose(1, 2)  # (B*T, num_patches, hidden_dim)

        # Pool patches to single frame representation
        pooled = self.pooling(patch_features)  # (B*T, 1, hidden_dim)
        pooled = pooled.squeeze(1)  # (B*T, hidden_dim)
        pooled = pooled.view(b, s, -1)  # (B, T, hidden_dim)

        # Compute temporal differences (frame velocity features)
        diffs = torch.diff(pooled, dim=1)
        diffs = torch.cat([pooled[:, 0:1, :], diffs], dim=1)

        # Concatenate position and velocity features
        features = torch.cat([pooled, diffs], dim=-1)  # (B, T, 2*hidden_dim)

        # Apply temporal head
        head_output = self.head(features)  # (B, T, num_hid_units)

        # Classify each frame
        logits = self.classifier(head_output)  # (B, T, num_classes)
        probabilities = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': features,
        }

    def get_backbone_info(self) -> dict[str, Any]:
        """Get information about the current backbone configuration.

        Returns:
            Dict with backbone type, hidden size, and other metadata.
        """
        return {
            'backbone_type': self.backbone_type,
            'hidden_size': self.embed_dim,
            'patch_size': self.backbone.patch_size,
            'frozen': self.freeze_backbone,
        }
