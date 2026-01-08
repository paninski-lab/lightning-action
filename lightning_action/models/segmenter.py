"""Action segmentation models with Lightning integration.

This module contains the main segmentation models adapted from daart but
updated to use PyTorch Lightning for training and modern architectural patterns.
"""

import logging
from abc import abstractmethod
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torchmetrics import Accuracy, F1Score
from typeguard import typechecked

from lightning_action.data.utils import compute_sequence_pad


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base Lightning model for action segmentation.
    
    This class provides the Lightning infrastructure and common functionality
    for segmentation models. Inherit from this class to create specific
    segmentation architectures.
    """

    @typechecked
    def __init__(self, config: dict[str, Any]):
        """Initialize base model.
        
        Args:
            config: configuration dictionary with model, optimizer, and training settings
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # extract model configuration
        self.model_config = config.get('model', {})
        self.input_size = self.model_config['input_size']
        self.output_size = self.model_config['output_size']
        self.sequence_length = self.model_config.get('sequence_length', 500)

        # ignore index
        self.ignore_index = config.get('data', {}).get('ignore_index', -100)

        # set random seed for reproducibility
        if 'seed' in self.model_config:
            pl.seed_everything(self.model_config['seed'])

        # initialize metrics
        self._setup_metrics()
        
        # build model architecture (implemented by subclasses)
        self._build_model()

        # compute sequence padding (after _build_model to make it easier to test unsupported backbones)
        self.sequence_pad = compute_sequence_pad(config['model']['backbone'], **config['model'])

    def _setup_metrics(self):
        """Set up torchmetrics for evaluation."""
        num_classes = self.output_size

        # training metrics
        self.train_accuracy = Accuracy(
            task='multiclass', num_classes=num_classes, ignore_index=self.ignore_index,
        )
        self.train_f1 = F1Score(
            task='multiclass', num_classes=num_classes, ignore_index=self.ignore_index,
        )

        # validation metrics
        self.val_accuracy = Accuracy(
            task='multiclass', num_classes=num_classes, ignore_index=self.ignore_index,
        )
        self.val_f1 = F1Score(
            task='multiclass', num_classes=num_classes, ignore_index=self.ignore_index,
        )

    @abstractmethod
    def _build_model(self):
        """Build the model architecture. Implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: input tensor with shape (batch, sequence, features)
            
        Returns:
            dictionary with model outputs including 'logits' and 'probabilities'
        """
        raise NotImplementedError

    def _remove_padding(
        self,
        data: dict[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Remove padding from each sequence for convolution/rnn models"""
        if self.sequence_pad is None or self.sequence_pad == 0:
            return data

        if isinstance(data, dict):
            for key, val in data.items():
                data[key] = data[key][:, self.sequence_pad:-self.sequence_pad]
        else:
            data = data[:, self.sequence_pad:-self.sequence_pad]

        return data

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        stage: str = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and metrics.
        
        Args:
            outputs: model outputs dictionary containing 'logits' and 'probabilities'
            targets: ground truth labels, either:
                - One-hot encoded: shape (batch, sequence, num_classes)
                - Class indices: shape (batch, sequence)
            stage: training stage ('train', 'val', 'test')
            
        Returns:
            tuple of (loss tensor, metrics dictionary)
        """
        logits = outputs['logits']

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.output_size)
        
        # Handle both one-hot and class index targets
        if targets.dim() == 3 and targets.shape[-1] == self.output_size:
            # One-hot encoded: (batch, sequence, num_classes) -> class indices
            targets_flat = torch.argmax(targets.reshape(-1, self.output_size), dim=-1)
        else:
            # Already class indices: (batch, sequence)
            targets_flat = targets.reshape(-1)

        # Handle edge case where all targets are ignore_index
        if torch.all(targets_flat == self.ignore_index):
            # Loss is 0, metrics are meaningless so set to 1.0
            loss = logits_flat.sum() * 0.0
            return loss, {
                f'{stage}_loss': 0.0,
                f'{stage}_accuracy': float('nan'),
                f'{stage}_f1': float('nan'),
            }

        # Get class weights from config and move to the correct device
        class_weights = self.model_config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=self.device, dtype=torch.float)
            
        # Compute cross entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            weight=class_weights,
        )

        # Compute metrics
        with torch.no_grad():
            pred_classes = torch.argmax(logits_flat, dim=-1)

            if stage == 'train':
                accuracy = self.train_accuracy(pred_classes, targets_flat)
                f1 = self.train_f1(pred_classes, targets_flat)
            else:  # val or test
                accuracy = self.val_accuracy(pred_classes, targets_flat)
                f1 = self.val_f1(pred_classes, targets_flat)

        # Handle NaN values to avoid contaminating epoch-level logging
        metrics = {}
        if not torch.isnan(loss):
            metrics[f'{stage}_loss'] = loss.item()
        if not torch.isnan(torch.tensor(accuracy)):
            metrics[f'{stage}_accuracy'] = accuracy.item()
        if not torch.isnan(torch.tensor(f1)):
            metrics[f'{stage}_f1'] = f1.item()
        
        return loss, metrics

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: batch dictionary with input data and targets
            batch_idx: batch index
            
        Returns:
            loss tensor
        """
        # get inputs and targets
        x = batch['input']
        targets = batch['labels']
        
        # forward pass
        outputs = self.forward(x)

        # remove padding
        outputs_no_pad = self._remove_padding(outputs)
        targets_no_pad = self._remove_padding(targets)

        # compute loss and metrics
        loss, metrics = self.compute_loss(outputs_no_pad, targets_no_pad, stage='train')
        
        # log metrics (only if we have valid metrics to log)
        if metrics:  # will be empty if all metrics were NaN
            self.log_dict(
                metrics,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=x.shape[0],
            )

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step.
        
        Args:
            batch: batch dictionary with input data and targets
            batch_idx: batch index
        """
        # get inputs and targets
        x = batch['input']
        targets = batch['labels']
        
        # forward pass
        outputs = self.forward(x)

        # remove padding
        outputs_no_pad = self._remove_padding(outputs)
        targets_no_pad = self._remove_padding(targets)

        # compute loss and metrics
        loss, metrics = self.compute_loss(outputs_no_pad, targets_no_pad, stage='val')

        # log metrics (only if we have valid metrics to log)
        if metrics:  # will be empty if all metrics were NaN
            self.log_dict(
                metrics,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=x.shape[0],
            )

        return None

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Prediction step.
        
        Args:
            batch: batch dictionary with input data
            batch_idx: batch index
            
        Returns:
            dictionary with predictions
        """
        # get inputs
        x = batch['input']
        
        # forward pass
        outputs = self.forward(x)

        # remove padding
        outputs_no_pad = self._remove_padding(outputs)

        # return predictions
        return {
            'logits': outputs_no_pad['logits'],
            'probabilities': outputs_no_pad['probabilities'],
            'predictions': torch.argmax(outputs_no_pad['probabilities'], dim=-1),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Supports:
        - Optimizers: Adam, AdamW, SGD
        - Schedulers: step, cosine, cosine_warm_restarts, reduce_on_plateau
        
        Config structure:
            optimizer:
                type: Adam  # or AdamW, SGD
                lr: 0.001
                wd: 0.0  # weight decay
                momentum: 0.9  # for SGD only
                scheduler:
                    use_scheduler: true
                    type: cosine  # or step, cosine_warm_restarts, reduce_on_plateau
                    # Scheduler-specific params (T_max, step_size, etc.)
        
        Returns:
            optimizer 
        """
        optimizer_config = self.config.get('optimizer', {})
        
        # Optimizer settings
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = float(optimizer_config.get('lr', 1e-3))
        weight_decay = float(optimizer_config.get('wd', 0.1))
        
        # Get parameters to optimize (allows subclass customization)
        params = self._get_optimizer_params()
        
        # Create optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = float(optimizer_config.get('momentum', 0.9))
            optimizer = torch.optim.SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')
        
        # Parse scheduler config (support both flat and nested structures)
        scheduler_config = optimizer_config.get('scheduler', None)
        
        # Handle flat scheduler specification: scheduler: 'cosine'
        if isinstance(scheduler_config, str):
            scheduler_type = scheduler_config.lower()
            scheduler_config = {}
        elif scheduler_config is None:
            # No scheduler
            return {'optimizer': optimizer}
        elif not scheduler_config.get('use_scheduler', True):
            # Scheduler explicitly disabled
            return {'optimizer': optimizer}
        else:
            scheduler_type = scheduler_config.get('type', 'cosine').lower()
        
        # Create scheduler
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1),
            )
        
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', optimizer_config.get('T_max', 100))
            eta_min_factor = scheduler_config.get('eta_min_factor', 20)
            eta_min = lr / eta_min_factor if eta_min_factor else 0
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        
        elif scheduler_type in ['cosine_warm_restarts', 'cosineannealingwarmrestarts']:
            T_0 = scheduler_config.get('T_0', 34)
            T_mult = scheduler_config.get('T_mult', 2)
            eta_min_factor = scheduler_config.get('eta_min_factor', 20)
            eta_min = lr / eta_min_factor if eta_min_factor else 0
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
        
        elif scheduler_type in ['reduce_on_plateau', 'reducelronplateau']:
            factor = scheduler_config.get('factor', 0.5)
            patience = scheduler_config.get('patience', 10)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Pointless for most schedulers
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
                    'monitor': 'val_loss',  # Pointless for most schedulers
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

    def _get_optimizer_params(self):
        """Get parameters to optimize.
        
        Override in subclasses for custom parameter groups.
        Default returns self.parameters().
        
        Returns:
            Parameters or list of parameter group dicts.
        """
        return self.parameters()


class Segmenter(BaseModel):
    """Main segmentation model for action recognition.
    
    This model implements supervised action segmentation using a flexible
    backbone architecture with a classification head.
    """

    def _build_model(self):
        """Build the segmentation model architecture."""
        # build backbone network
        self.backbone = self._build_backbone()
        
        # build classification head
        backbone_output_size = self._get_backbone_output_size()
        self.classifier = nn.Linear(backbone_output_size, self.output_size)
        
        # initialize weights
        self._initialize_weights()

    def _build_backbone(self) -> nn.Module:
        """Build the backbone network.
        
        Returns:
            backbone network module
        """
        backbone_type = self.model_config.get('backbone', 'temporalmlp')

        logger.info(f'Contructing Segmenter model with {backbone_type} backbone')
        
        if backbone_type.lower() == 'temporalmlp':
            from lightning_action.models.backbones import TemporalMLP
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
            from lightning_action.models.backbones import RNN
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
            from lightning_action.models.backbones import DilatedTCN
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
        """Get the output size of the backbone network.
        
        Returns:
            output feature size of the backbone
        """
        # both TemporalMLP and RNN output num_hid_units features
        return self.model_config['num_hid_units']

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the segmentation model.
        
        Args:
            x: input tensor with shape (batch, sequence, features)
            
        Returns:
            dictionary with 'logits' and 'probabilities'
        """
        batch_size, sequence_length, features = x.shape

        # pass through backbone
        backbone_features = self.backbone(x)

        # classify each time step
        logits = self.classifier(backbone_features)

        # compute probabilities
        probabilities = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': backbone_features.view(batch_size, sequence_length, -1),
        }
