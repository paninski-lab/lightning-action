"""Video segmentation model with ViT-MAE backbone and temporal modeling.

This module implements a video action segmentation model that:
1. Encodes individual frames using a ViT-MAE (Vision Transformer with Masked Autoencoding)
2. Pools spatial features from each frame into a single vector
3. Models temporal dynamics using a backbone network (TCN, MLP, or RNN)
4. Classifies each frame into an action class

VideoBaseModel inherits from BaseModel (models/segmenter.py) and overrides
only the methods that need video-specific behavior.
"""

import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from lightning_action.models.segmenter import BaseModel
from lightning_action.models.encoders.vitmae import ImageEncoderViTMAE
from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
from lightning_action.models.backbones import DilatedTCN, TemporalMLP, RNN


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
        batch: Union[Tuple, dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[dict]]]:
        """Extract inputs and targets from batch.
        
        Handles both tuple format (from DALI) and dict format (from standard loaders).
        
        Args:
            batch: Either tuple (frames, labels, metadata) or dict with 'input'/'labels'.
        
        Returns:
            Tuple of (inputs, targets, metadata).
        """
        if isinstance(batch, dict):
            # Standard dataloader format
            return batch['input'], batch.get('labels'), None
        
        # DALI tuple format: (frames, labels, metadata) or (frames, labels)
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
        batch: Union[Tuple[torch.Tensor, torch.Tensor, List[dict]], dict],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
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
        frames, labels, _ = self._get_inputs_and_targets(batch)
        
        # Check if all labels are ignored (after trimming padding)
        if self.sequence_pad and self.sequence_pad > 0:
            trimmed_labels = labels[:, self.sequence_pad:-self.sequence_pad]
        else:
            trimmed_labels = labels
        
        all_ignored = torch.all(trimmed_labels == self.ignore_index)
        
        # For DDP: coordinate skip decision across all GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            all_ignored_tensor = all_ignored.float().unsqueeze(0)
            all_ignored_list = [torch.zeros(1, device=self.device) for _ in range(world_size)]
            torch.distributed.all_gather(all_ignored_list, all_ignored_tensor)
            all_ignored_gathered = torch.cat(all_ignored_list)
            
            # Skip only if ALL GPUs have all-ignore batches
            if torch.all(all_ignored_gathered > 0.5):
                return None
        elif all_ignored:
            return None
        
        # Forward pass
        outputs = self.forward(frames)
        
        # Remove padding
        outputs_no_pad = self._remove_padding(outputs)
        labels_no_pad = self._remove_padding(labels)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(outputs_no_pad, labels_no_pad, stage='train')
        
        # Log metrics
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
        batch: Union[Tuple[torch.Tensor, torch.Tensor, List[dict]], dict],
        batch_idx: int,
    ) -> None:
        """Execute one validation step.
        
        Args:
            batch: Tuple of (frames, labels, metadata) or dict.
            batch_idx: Index of this batch.
        """
        frames, labels, _ = self._get_inputs_and_targets(batch)
        
        # Forward pass
        outputs = self.forward(frames)
        
        # Remove padding
        outputs_no_pad = self._remove_padding(outputs)
        labels_no_pad = self._remove_padding(labels)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(outputs_no_pad, labels_no_pad, stage='val')
        
        # Log metrics
        if metrics:
            self.log_dict(
                metrics,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=frames.shape[0],
            )

    @typechecked
    def predict_step(
        self,
        batch: Union[Tuple[torch.Tensor, List[int], List[dict]], dict],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Execute one prediction step with boundary-aware output slicing.
        
        The extended sequence from DALI provides context on both sides.
        For boundary chunks (first/last in a video), we adjust slicing to
        include predictions for frames that would normally be cut off:
        
        - Start chunk: include predictions from frame 0 (use [0:-pad] slice)
        - Middle chunk: standard [pad:-pad] slice  
        - End chunk: include predictions to video end (use [pad:] slice)
        
        Args:
            batch: Tuple of (frames, actual_lengths, metadata) or dict.
            batch_idx: Index of this batch.
            dataloader_idx: Index of dataloader (for multiple dataloaders).
        
        Returns:
            List of probability tensors, one per sample in batch.
        """
        frames, _, metadata = self._get_inputs_and_targets(batch)
        
        # Extract boundary flags from metadata
        is_start = None
        is_end = None
        if metadata is not None:
            is_start = [m.get('is_start', False) for m in metadata]
            is_end = [m.get('is_end', False) for m in metadata]
        
        # Forward pass
        outputs = self.forward(frames)
        probabilities = outputs['probabilities']  # (B, extended_seq, num_classes)
        
        result = []
        pad = self.sequence_pad or 0
        
        for i in range(probabilities.shape[0]):
            sample_probs = probabilities[i]  # (extended_seq, num_classes)
            sample_is_start = is_start[i] if is_start else False
            sample_is_end = is_end[i] if is_end else False
            
            # Adjust slicing based on chunk position in video
            if pad > 0:
                if sample_is_start and sample_is_end:
                    # Single chunk covers entire video - use standard slice
                    valid_probs = sample_probs[pad:-pad]
                elif sample_is_start:
                    # First chunk - include predictions from frame 0
                    valid_probs = sample_probs[:-pad]
                elif sample_is_end:
                    # Last chunk - include predictions to end
                    valid_probs = sample_probs[pad:]
                else:
                    # Middle chunk - standard trimming
                    valid_probs = sample_probs[pad:-pad]
            else:
                # No padding - use all predictions
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
    """Video action segmentation model with ViT-MAE backbone.
    
    This model processes video frames through:
    1. ViT-MAE encoder: Extract patch-level features from each frame
    2. Attention pooling: Aggregate patches into frame-level features
    3. Feature augmentation: Concatenate positions with velocities
    4. Temporal backbone: Model temporal dynamics
    5. Classifier: Predict action class per frame
    
    The model supports optional encoder freezing for transfer learning
    and can fine-tune just the last encoder layer if desired.
    
    Attributes:
        encoder: ViT-MAE image encoder.
        pooling: MultiheadAttentionPooling for spatial pooling.
        backbone: Temporal sequence model.
        classifier: Linear classification head.
    """

    def _build_model(self) -> None:
        """Build the complete video segmentation model."""
        # Initialize encoder
        self.encoder = ImageEncoderViTMAE()
        
        encoder_ckpt = self.model_config.get('encoder_checkpoint')
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            self.encoder.load_pretrained_weights(encoder_ckpt)
        
        self.embed_dim = self.encoder.hidden_size
        
        # Configure encoder freezing
        self.freeze_encoder = self.model_config.get('freeze_encoder', True)
        
        if not self.freeze_encoder:
            # Partial fine-tuning: freeze all except last layer and layernorm
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze last transformer layer
            for param in self.encoder.vit_mae.encoder.layer[-1].parameters():
                param.requires_grad = True
            
            # Unfreeze final layer norm
            for param in self.encoder.vit_mae.layernorm.parameters():
                param.requires_grad = True
        else:
            # Full freeze: no encoder gradients
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Spatial pooling using learned queries
        self.pooling = MultiheadAttentionPooling(
            embed_dim=self.embed_dim,
            num_heads=8,
            num_seeds=1,
            dropout=0.0,
            use_ffn=True,
            layer_norm=False,
        )
        
        # Build temporal backbone
        self.backbone = self._build_backbone()
        
        # Classification head
        backbone_output_size = self._get_backbone_output_size()
        self.classifier = nn.Linear(backbone_output_size, self.output_size)
        
        # Initialize weights (excluding encoder)
        self._initialize_weights()

    def _build_backbone(self) -> nn.Module:
        """Construct temporal backbone for sequence modeling.
        
        Returns:
            Configured temporal backbone module.
        
        Raises:
            ValueError: If backbone type is not supported.
        """
        backbone_type = self.model_config.get('backbone', 'dtcn')
        
        if backbone_type.lower() == 'temporalmlp':
            return TemporalMLP(
                input_size=self.input_size,
                num_hid_units=self.model_config['num_hid_units'],
                num_layers=self.model_config['num_layers'],
                num_lags=self.model_config.get('num_lags', 1),
                activation=self.model_config.get('activation', 'lrelu'),
                dropout_rate=self.model_config.get('dropout_rate', 0.0),
                seed=self.model_config.get('seed', 42),
            )
        elif backbone_type.lower() == 'rnn':
            return RNN(
                input_size=self.input_size,
                num_hid_units=self.model_config['num_hid_units'],
                num_layers=self.model_config['num_layers'],
                rnn_type=self.model_config.get('rnn_type', 'lstm'),
                bidirectional=self.model_config.get('bidirectional', False),
                dropout_rate=self.model_config.get('dropout_rate', 0.0),
                seed=self.model_config.get('seed', 42),
            )
        elif backbone_type.lower() in ['dtcn', 'dilatedtcn']:
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
            raise ValueError(f'Unsupported backbone type: {backbone_type}')

    def _get_backbone_output_size(self) -> int:
        """Get the output feature dimension of the backbone."""
        return self.model_config['num_hid_units']

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.pooling, self.backbone, self.classifier]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def _get_optimizer_params(self):
        """Get parameters to optimize with specific groups.
        
        Returns parameters from pooling, backbone, and classifier.
        Encoder parameters are handled separately (frozen or partially frozen).
        
        Returns:
            List of parameter group dicts.
        """
        return [
            {'params': self.pooling.parameters()},
            {'params': self.backbone.parameters()},
            {'params': self.classifier.parameters()},
        ]

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence channels height width'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through video segmentation model.
        
        Processing steps:
        1. Reshape for batch encoding: (B, T, C, H, W) -> (B*T, C, H, W)
        2. ViT-MAE encoding: Extract patch features
        3. Spatial pooling: Aggregate patches per frame
        4. Feature augmentation: Add temporal differences (velocity)
        5. Temporal backbone: Model dynamics
        6. Classification: Per-frame action prediction
        
        Args:
            x: Input video frames, shape (batch, time, channels, height, width).
        
        Returns:
            Dict containing:
            - 'logits': Raw class scores, shape (B, T, num_classes)
            - 'probabilities': Softmax probabilities, shape (B, T, num_classes)
            - 'features': Frame features before backbone, shape (B, T, 2*embed_dim)
        """
        b, s, c, h, w = x.shape
        
        # Flatten batch and time for frame-wise encoding
        x = x.view(b * s, c, h, w)
        
        # Encode frames with ViT-MAE
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder:
                self.encoder.eval()
            patch_features = self.encoder(x)  # (B*T, hidden_dim, H', W')
        
        # Reshape for attention pooling: (B*T, H'*W', hidden_dim)
        bs, feat_d, feat_h, feat_w = patch_features.shape
        num_patches = feat_h * feat_w
        patch_features = patch_features.reshape(bs, feat_d, num_patches)
        patch_features = patch_features.transpose(1, 2)
        
        # Pool patches to single frame representation
        pooled = self.pooling(patch_features)  # (B*T, 1, hidden_dim)
        pooled = pooled.squeeze(1)  # (B*T, hidden_dim)
        pooled = pooled.view(b, s, -1)  # (B, T, hidden_dim)

        # Compute temporal differences (frame velocity features)
        # This helps the model understand motion/change
        diffs = torch.diff(pooled, dim=1)
        diffs = torch.cat([pooled[:, 0:1, :], diffs], dim=1)  # Pad first frame
        
        # Concatenate position and velocity features
        features = torch.cat([pooled, diffs], dim=-1)  # (B, T, 2*hidden_dim)
        
        # Apply temporal backbone
        backbone_output = self.backbone(features)  # (B, T, num_hid_units)
        
        # Classify each frame
        logits = self.classifier(backbone_output)  # (B, T, num_classes)
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': features,
        }
