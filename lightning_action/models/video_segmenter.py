"""Video segmentation model with ViT-MAE backbone and temporal modeling.

This module implements a video action segmentation model that:
1. Encodes individual frames using a ViT-MAE (Vision Transformer with Masked Autoencoding)
2. Pools spatial features from each frame into a single vector
3. Models temporal dynamics using a backbone network (TCN, MLP, or RNN)
4. Classifies each frame into an action class

Architecture Overview:
    Input: (B, T, C, H, W) video frames
    
    1. Frame Encoding (per-frame):
       - ViT-MAE: (B*T, C, H, W) -> (B*T, num_patches, hidden_dim)
       
    2. Spatial Pooling (per-frame):
       - PMA: (B*T, num_patches, hidden_dim) -> (B*T, 1, hidden_dim)
       
    3. Temporal Modeling:
       - Reshape: (B, T, hidden_dim)
       - Add frame differences for velocity features
       - Backbone: (B, T, 2*hidden_dim) -> (B, T, num_hid_units)
       
    4. Classification:
       - Linear: (B, T, num_hid_units) -> (B, T, num_classes)

Supported temporal backbones:
- DilatedTCN: Dilated temporal convolutions with exponentially growing receptive field
- TemporalMLP: Simple MLP with sliding window
- RNN: LSTM or GRU, optionally bidirectional

Example usage:
    config = {
        'model': {
            'input_size': 1536,
            'output_size': 3,
            'backbone': 'dtcn',
            'num_layers': 4,
            'num_hid_units': 32,
            'num_lags': 2,
        },
        'data': {'ignore_index': -100},
        'optimizer': {'lr': 0.001, 'type': 'Adam'},
    }
    model = VideoSegmenter(config)
"""
import math
import os
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torchmetrics import Accuracy, F1Score
from typeguard import typechecked

from lightning_action.models.encoders.vitmae import ImageEncoderViTMAE
from lightning_action.models.backbones import DilatedTCN, TemporalMLP, RNN
from lightning_action.data.utils import compute_sequence_pad
from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling



class VideoBaseModel(pl.LightningModule):
    """Base Lightning model for video action segmentation.
    
    Provides common functionality for all video segmentation models:
    - Metric tracking (accuracy, F1)
    - Loss computation with class weighting
    - Training/validation step implementations
    - Optimizer configuration with scheduler support
    
    Subclasses must implement:
    - _build_model(): Construct the model architecture
    - forward(): Define the forward pass
    
    Attributes:
        config: Full configuration dictionary.
        model_config: Model-specific configuration.
        input_size: Input feature dimension.
        output_size: Number of output classes.
        ignore_index: Label value to ignore in loss/metrics.
    """

    @typechecked
    def __init__(self, config: dict[str, Any]):
        """Initialize the base video model.
        
        Args:
            config: Configuration dictionary containing:
                - model: Model architecture settings
                - data: Data settings including ignore_index
                - optimizer: Optimizer and scheduler settings
                - training: Training settings
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Extract model configuration
        self.model_config = config.get('model', {})
        self.input_size = self.model_config['input_size']
        self.output_size = self.model_config['output_size']
        self.sequence_length = self.model_config.get('sequence_length', 128)
        self.ignore_index = config.get('data', {}).get('ignore_index', -100)

        # Set random seed if specified
        if 'seed' in self.model_config:
            pl.seed_everything(self.model_config['seed'])

        self._setup_metrics()
        self._build_model()

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for tracking performance.
        
        Creates separate metric instances for train and validation
        to avoid metric state contamination.
        """
        num_classes = self.output_size

        # Training metrics
        self.train_accuracy = Accuracy(
            task='multiclass', 
            num_classes=num_classes, 
            ignore_index=self.ignore_index,
        )
        self.train_f1 = F1Score(
            task='multiclass', 
            num_classes=num_classes, 
            ignore_index=self.ignore_index,
        )

        # Validation metrics (separate instances)
        self.val_accuracy = Accuracy(
            task='multiclass', 
            num_classes=num_classes, 
            ignore_index=self.ignore_index,
        )
        self.val_f1 = F1Score(
            task='multiclass', 
            num_classes=num_classes, 
            ignore_index=self.ignore_index,
        )

    @abstractmethod
    def _build_model(self) -> None:
        """Build the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    @typechecked
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Int[torch.Tensor, 'batch sequence'],
        stage: str = 'train',
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss and metrics for a batch.
        
        Handles TCN padding by trimming predictions and targets to the
        "middle" region where we have full temporal context.
        
        Args:
            outputs: Model outputs dict containing 'logits'.
            targets: Ground truth labels, shape (B, T).
            stage: Either 'train' or 'val' for metric selection.
        
        Returns:
            Tuple of (loss, metrics_dict).
        """
        logits = outputs['logits']
        
        # Trim TCN padding regions (model can't make good predictions there)
        if hasattr(self, 'tcn_padding') and self.tcn_padding > 0:
            logits = logits[:, self.tcn_padding:-self.tcn_padding, :]
            targets = targets[:, self.tcn_padding:-self.tcn_padding]
        
        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, self.output_size)
        targets_flat = targets.reshape(-1)

        # Handle edge case where all targets are ignore_index
        # This can happen with certain data splits
        if torch.all(targets_flat == self.ignore_index):
            # Return zero loss (but maintain gradient graph)
            loss = logits_flat.sum() * 0.0
            
            with torch.no_grad():
                pred_classes = torch.argmax(logits_flat, dim=-1)
                if stage == 'train':
                    accuracy = self.train_accuracy(pred_classes, targets_flat)
                    f1 = self.train_f1(pred_classes, targets_flat)
                else:
                    accuracy = self.val_accuracy(pred_classes, targets_flat)
                    f1 = self.val_f1(pred_classes, targets_flat)
            
            metrics = {
                f'{stage}_loss': loss,
                f'{stage}_accuracy': accuracy,
                f'{stage}_f1': f1,
            }
            return loss, metrics

        # Get optional class weights for handling imbalanced data
        class_weights = self.model_config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(
                class_weights, device=self.device, dtype=torch.float
            )
            
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            weight=class_weights,
        )

        # Compute metrics (no gradients needed)
        with torch.no_grad():
            pred_classes = torch.argmax(logits_flat, dim=-1)

            if stage == 'train':
                accuracy = self.train_accuracy(pred_classes, targets_flat)
                f1 = self.train_f1(pred_classes, targets_flat)
            else:
                accuracy = self.val_accuracy(pred_classes, targets_flat)
                f1 = self.val_f1(pred_classes, targets_flat)

        metrics = {
            f'{stage}_loss': loss,
            f'{stage}_accuracy': accuracy,
            f'{stage}_f1': f1,
        }
        
        return loss, metrics

    @typechecked
    def training_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor, List[dict]], List[torch.Tensor]],
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """Execute one training step.
        
        Handles the special case where all labels in a batch are ignore_index
        by skipping the batch (returns None). For DDP, coordinates across
        all GPUs to skip only if ALL GPUs have all-ignore batches.
        
        Args:
            batch: Tuple of (frames, labels, metadata).
            batch_idx: Index of this batch.
        
        Returns:
            Loss tensor, or None if batch should be skipped.
        """
        frames, labels, _ = batch
        
        # Check if all labels are ignored (after trimming padding)
        if hasattr(self, 'tcn_padding') and self.tcn_padding > 0:
            trimmed_labels = labels[:, self.tcn_padding:-self.tcn_padding]
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
        
        # Standard forward and loss computation
        outputs = self.forward(frames)
        loss, metrics = self.compute_loss(outputs, labels, stage='train')
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_f1', self.train_f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    @typechecked
    def validation_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor, List[dict]], List[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Execute one validation step.
        
        Args:
            batch: Tuple of (frames, labels, metadata) or (frames, labels).
            batch_idx: Index of this batch.
        """
        if len(batch) == 3:
            frames, labels, _ = batch
        else:
            frames, labels = batch
        
        outputs = self.forward(frames)
        loss, metrics = self.compute_loss(outputs, labels, stage='val')
        
        # Log validation metrics
        self.log('val_loss', metrics['val_loss'], 
                 prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_accuracy', metrics['val_accuracy'], 
                 prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_f1', metrics['val_f1'], 
                 prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    @typechecked
    def predict_step(
        self,
        batch: Union[Tuple[torch.Tensor, List[int], List[dict]], List[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Execute one prediction step with boundary-aware output slicing.
        
        The extended sequence from DALI provides TCN context on both sides.
        For boundary chunks (first/last in a video), we adjust slicing to
        include predictions for frames that would normally be cut off:
        
        - Start chunk: include predictions from frame 0 (use [0:-pad] slice)
        - Middle chunk: standard [pad:-pad] slice  
        - End chunk: include predictions to video end (use [pad:] slice)
        
        Args:
            batch: Tuple of (frames, actual_lengths, metadata).
            batch_idx: Index of this batch.
            dataloader_idx: Index of dataloader (for multiple dataloaders).
        
        Returns:
            List of probability tensors, one per sample in batch.
        """
        if len(batch) == 3:
            frames, actual_lengths, metadata = batch
        else:
            frames, actual_lengths = batch
            metadata = None
        
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
        for i in range(probabilities.shape[0]):
            sample_probs = probabilities[i]  # (extended_seq, num_classes)
            sample_is_start = is_start[i] if is_start else False
            sample_is_end = is_end[i] if is_end else False
            
            # Adjust slicing based on chunk position in video
            if self.tcn_padding > 0:
                if sample_is_start and sample_is_end:
                    # Single chunk covers entire video - use standard slice
                    valid_probs = sample_probs[self.tcn_padding:-self.tcn_padding]
                elif sample_is_start:
                    # First chunk - include predictions from frame 0
                    valid_probs = sample_probs[:-self.tcn_padding]
                elif sample_is_end:
                    # Last chunk - include predictions to end
                    valid_probs = sample_probs[self.tcn_padding:]
                else:
                    # Middle chunk - standard trimming
                    valid_probs = sample_probs[self.tcn_padding:-self.tcn_padding]
            else:
                # No TCN padding - use all predictions
                valid_probs = sample_probs
            
            result.append(valid_probs)
        
        return result

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch. Clears GPU cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch. Clears GPU cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch. Clears GPU cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @typechecked
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.
        
        Supports:
        - Optimizers: Adam, AdamW, SGD
        - Schedulers: CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
        
        Returns:
            Dict with 'optimizer' and optionally 'lr_scheduler'.
        """
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('wd', 0.0)
        
        # Collect parameters from all trainable components
        trainable_params = [
            {'params': self.pooling.parameters()},
            {'params': self.backbone.parameters()},
            {'params': self.classifier.parameters()},
        ]
        
        # Create optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                trainable_params, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

        # Configure learning rate scheduler
        scheduler_config = optimizer_config.get('scheduler', {})
        use_scheduler = scheduler_config.get('use_scheduler', False)
        
        if use_scheduler:
            scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
            
            if scheduler_type == 'CosineAnnealingWarmRestarts':
                # Warm restarts: restart LR schedule periodically
                T_0 = scheduler_config.get('T_0', 34)
                T_mult = scheduler_config.get('T_mult', 2)
                eta_min_factor = scheduler_config.get('eta_min_factor', 20)
                eta_min = lr / eta_min_factor
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
                )
                
            elif scheduler_type == 'CosineAnnealingLR':
                # Single cosine decay
                T_max = optimizer_config.get('T_max', 200)
                eta_min_factor = scheduler_config.get('eta_min_factor', 20)
                eta_min = lr / eta_min_factor
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=eta_min
                )
                
            elif scheduler_type == 'ReduceLROnPlateau':
                # Reduce LR when validation loss plateaus
                factor = scheduler_config.get('factor', 0.5)
                patience = scheduler_config.get('patience', 10)
                
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=factor, patience=patience, verbose=True
                )
                
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1,
                    }
                }
            else:
                raise ValueError(f'Unsupported scheduler type: {scheduler_type}')
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        return optimizer



class VideoSegmenter(VideoBaseModel):
    """Video action segmentation model with ViT-MAE backbone.
    
    This model processes video frames through:
    1. ViT-MAE encoder: Extract patch-level features from each frame
    2. PMA pooling: Aggregate patches into frame-level features
    3. Feature augmentation: Concatenate positions with velocities
    4. Temporal backbone: Model temporal dynamics
    5. Classifier: Predict action class per frame
    
    The model supports optional encoder freezing for transfer learning
    and can fine-tune just the last encoder layer if desired.
    
    Attributes:
        encoder: ViT-MAE image encoder.
        pooling: PMA for spatial pooling.
        backbone: Temporal sequence model.
        classifier: Linear classification head.
        tcn_padding: Required padding for temporal receptive field.
    """
    
    @typechecked
    def __init__(self, config: dict[str, Any]):
        """Initialize the VideoSegmenter.
        
        Args:
            config: Configuration dictionary containing model, data,
                optimizer, and training settings.
        """
        # Store num_lags before parent init (needed for padding calculation)
        self.num_lags = config.get('model', {}).get('num_lags', 0)
        super().__init__(config)
        
        # Initialize encoder (config comes from pretrained model)
        self.encoder = ImageEncoderViTMAE()
        
        # Load custom pretrained weights if available
        encoder_ckpt = self.model_config.get('encoder_checkpoint')
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            self.encoder.load_pretrained_weights(encoder_ckpt)
        
        # Get embed_dim from the encoder's config
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
            num_seeds=1,  # Pool to single vector per frame
            dropout=0.0,
            use_ffn=True,  # Matches original MAB behavior
            layer_norm=False,
        )
        
        # Build temporal backbone and classifier
        self._build_model()

    def _build_model(self) -> None:
        """Build temporal backbone and classification head."""
        self.backbone = self._build_backbone()
        backbone_output_size = self._get_backbone_output_size()
        self.classifier = nn.Linear(backbone_output_size, self.output_size)
        self._initialize_weights()
        self.tcn_padding = compute_sequence_pad(
            model_type=self.model_config.get('backbone', 'dtcn'),
            num_lags=self.model_config.get('num_lags', 1),
            num_layers=self.model_config.get('num_layers', 4),
            default=0,
        )

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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
