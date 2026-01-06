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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from lightning_action.models.segmenter import BaseModel
from lightning_action.models.encoders.vitmae import ImageEncoderViTMAE
from lightning_action.models.encoders.resnet import ImageEncoderResNet
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
        batch: Union[Tuple[torch.Tensor, torch.Tensor, List[dict]], dict],
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
        batch: Union[Tuple[torch.Tensor, List[int], List[dict]], dict],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
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
    """Video action segmentation model with swappable encoder.
    
    This model processes video frames through:
    1. Configurable encoder: Extract spatial features from each frame
       - ViT-MAE: Patch-level transformer features (hidden_size=768)
       - ResNet: Convolutional feature maps (hidden_size=512-2048)
    2. Attention pooling: Aggregate spatial features per frame
    3. Feature augmentation: Concatenate positions with velocities
    4. Temporal backbone: Model dynamics (TCN, MLP, or RNN)
    5. Classifier: Per-frame action prediction
    
    The encoder is selected via encoder_type in the model config section.
    
    Attributes:
        encoder: Image encoder (ViT-MAE or ResNet).
        encoder_type: String identifying the encoder type.
        pooling: MultiheadAttentionPooling for spatial pooling.
        backbone: Temporal sequence model.
        classifier: Linear classification head.
    """
    
    # Hidden sizes for each encoder type (used for input_size auto-computation)
    ENCODER_HIDDEN_SIZES = {
        'vitmae': 768,
        'vit-mae': 768,
        'vit': 768,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize VideoSegmenter with auto-computed input_size.
        
        Args:
            config: Configuration dictionary.
        """
        # Auto-compute input_size if not provided, before BaseModel.__init__ runs
        model_config = config.get('model', {})
        if model_config.get('input_size') is None:
            encoder_name = model_config.get('encoder', 'vitmae').lower()
            hidden_size = self.ENCODER_HIDDEN_SIZES.get(encoder_name, 768)
            # input_size = 2 * hidden_size (position + velocity features)
            config['model']['input_size'] = hidden_size * 2
        
        super().__init__(config)

    def _build_model(self) -> None:
        """Build the complete video segmentation model."""
        # Load encoder config from file if provided
        encoder_file_config = {}
        encoder_config_path = self.model_config.get('encoder_config_path')
        if encoder_config_path and os.path.exists(encoder_config_path):
            import yaml
            with open(encoder_config_path, 'r') as f:
                encoder_file_config = yaml.safe_load(f)
        
        # Extract model_params from encoder config (mirrors vit.yaml / resnet_ae.yaml structure)
        encoder_model_params = encoder_file_config.get('model', {}).get('model_params', {})
        
        # Get encoder name: main config 'encoder' overrides encoder_config_path
        encoder_name = self.model_config.get('encoder')
        if encoder_name is None:
            # Fall back to model_class or model_params.backbone from encoder config
            encoder_name = encoder_file_config.get('model', {}).get('model_class')
            if encoder_name == 'resnet':
                encoder_name = encoder_model_params.get('backbone', 'resnet50')
            elif encoder_name is None:
                encoder_name = 'vitmae'
        encoder_name = encoder_name.lower()
        
        # Build encoder based on name
        if encoder_name in ['vitmae', 'vit-mae', 'vit']:
            self.encoder_type = 'vitmae'
            vitmae_config = {
                # model_name at top level of encoder config
                'model_name': encoder_file_config.get('model_name',
                    self.model_config.get('encoder_model_name', 'facebook/vit-mae-base')),
                # mask_ratio in model.model_params
                'mask_ratio': encoder_model_params.get('mask_ratio',
                    self.model_config.get('encoder_mask_ratio', 0.0)),
            }
            self.encoder = ImageEncoderViTMAE(vitmae_config)
            
        elif encoder_name.startswith('resnet'):
            self.encoder_type = 'resnet'
            resnet_config = {
                'backbone': encoder_name,
                'image_size': encoder_model_params.get('image_size', 
                    self.model_config.get('encoder_image_size', 224)),
            }
            self.encoder = ImageEncoderResNet(resnet_config)
            
        else:
            raise ValueError(
                f"Unknown encoder: {encoder_name}. "
                f"Supported: 'vitmae', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'"
            )
        
        # Load encoder checkpoint if specified
        encoder_ckpt = self.model_config.get('encoder_checkpoint')
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            self.encoder.load_pretrained_weights(encoder_ckpt)
        
        # Get encoder hidden size
        self.embed_dim = self.encoder.hidden_size
        
        # Validate input_size matches encoder (it was auto-computed in __init__ if not set)
        expected_input_size = self.embed_dim * 2
        if self.input_size != expected_input_size:
            import warnings
            warnings.warn(
                f"Config input_size ({self.input_size}) differs from expected "
                f"({expected_input_size} = 2 × {self.embed_dim}). "
                f"Using config value."
            )
        
        # Configure encoder freezing
        self.freeze_encoder = self.model_config.get('freeze_encoder', True)
        self._configure_encoder_freezing()
        
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

    def _configure_encoder_freezing(self) -> None:
        """Configure encoder parameter freezing based on settings."""
        if self.freeze_encoder:
            # Full freeze: no encoder gradients
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Partial fine-tuning: freeze all except last layer
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze last layer using encoder's helper method
            for param in self.encoder.get_last_layer_params():
                param.requires_grad = True

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
        Encoder parameters are handled separately based on freeze settings.
        
        Returns:
            List of parameter group dicts.
        """
        param_groups = [
            {'params': self.pooling.parameters()},
            {'params': self.backbone.parameters()},
            {'params': self.classifier.parameters()},
        ]
        
        # Add encoder parameters if not fully frozen
        if not self.freeze_encoder:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                # Use lower learning rate for encoder fine-tuning
                encoder_lr = self.config.get('optimizer', {}).get('encoder_lr')
                if encoder_lr is None:
                    base_lr = self.config.get('optimizer', {}).get('lr', 1e-3)
                    encoder_lr = base_lr * 0.1
                
                param_groups.append({
                    'params': encoder_params,
                    'lr': encoder_lr,
                })
        
        return param_groups

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence channels height width'],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through video segmentation model.
        
        Processing steps:
        1. Reshape for batch encoding: (B, T, C, H, W) -> (B*T, C, H, W)
        2. Encoder: Extract spatial features (works for both ViT and ResNet)
        3. Spatial pooling: Aggregate to frame-level features
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
        
        # Encode frames with selected encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder:
                self.encoder.eval()
            spatial_features = self.encoder(x)  # (B*T, hidden_dim, H', W')
        
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
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about the current encoder configuration.
        
        Returns:
            Dict with encoder type, hidden size, and other metadata.
        """
        return {
            'encoder_type': self.encoder_type,
            'hidden_size': self.embed_dim,
            'patch_size': self.encoder.patch_size,
            'frozen': self.freeze_encoder,
        }
