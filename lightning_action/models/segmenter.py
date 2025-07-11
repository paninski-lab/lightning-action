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
        
        # set random seed for reproducibility
        if 'seed' in self.model_config:
            pl.seed_everything(self.model_config['seed'])
        
        # initialize metrics
        self._setup_metrics()
        
        # build model architecture (implemented by subclasses)
        self._build_model()

    def _setup_metrics(self):
        """Set up torchmetrics for evaluation."""
        num_classes = self.output_size
        
        # training metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        
        # validation metrics
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

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

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Int[torch.Tensor, 'batch sequence'],
        stage: str = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and metrics.
        
        Args:
            outputs: model outputs dictionary
            targets: ground truth labels
            stage: training stage ('train', 'val', 'test')
            
        Returns:
            tuple of (loss tensor, metrics dictionary)
        """
        logits = outputs['logits']
        
        # flatten for loss computation
        logits_flat = logits.view(-1, self.output_size)
        targets_flat = targets.view(-1)
        
        # compute cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # compute metrics
        with torch.no_grad():
            probabilities = outputs['probabilities']
            probs_flat = probabilities.view(-1, self.output_size)
            
            if stage == 'train':
                accuracy = self.train_accuracy(probs_flat, targets_flat)
                f1 = self.train_f1(probs_flat, targets_flat)
            else:  # val or test
                accuracy = self.val_accuracy(probs_flat, targets_flat)
                f1 = self.val_f1(probs_flat, targets_flat)
        
        metrics = {
            f'{stage}_loss': loss.item(),
            f'{stage}_accuracy': accuracy.item(),
            f'{stage}_f1': f1.item(),
        }
        
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
        x = batch['markers']  # or 'features' depending on signal type
        targets = batch['labels']
        
        # forward pass
        outputs = self.forward(x)
        
        # compute loss and metrics
        loss, metrics = self.compute_loss(outputs, targets, stage='train')
        
        # log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
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
        x = batch['markers']  # or 'features' depending on signal type
        targets = batch['labels']
        
        # forward pass
        outputs = self.forward(x)
        
        # compute loss and metrics
        loss, metrics = self.compute_loss(outputs, targets, stage='val')
        
        # log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        x = batch['markers']  # or 'features' depending on signal type
        
        # forward pass
        outputs = self.forward(x)
        
        # return predictions
        return {
            'logits': outputs['logits'],
            'probabilities': outputs['probabilities'],
            'predictions': torch.argmax(outputs['probabilities'], dim=-1),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            optimizer configuration dictionary
        """
        optimizer_config = self.config.get('optimizer', {})
        
        # default optimizer settings
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('wd', 0.0)
        
        # create optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')
        
        # setup scheduler if specified
        scheduler_type = optimizer_config.get('scheduler', None)
        if scheduler_type is None:
            return optimizer
        
        if scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=optimizer_config.get('step_size', 30),
                gamma=optimizer_config.get('gamma', 0.1),
            )
        elif scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=optimizer_config.get('T_max', 100),
            )
        else:
            raise ValueError(f'Unsupported scheduler type: {scheduler_type}')
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            },
        }


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
            
        Note:
            This is a placeholder implementation. In the future, this will
            create backbone networks from the backbones/ directory based on
            the model configuration.
        """
        # placeholder: simple MLP backbone
        hidden_size = self.model_config['num_hid_units']
        num_layers = self.model_config['num_layers']
        
        layers = []
        input_size = self.input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        return nn.Sequential(*layers)

    def _get_backbone_output_size(self) -> int:
        """Get the output size of the backbone network.
        
        Returns:
            output feature size of the backbone
        """
        # for the placeholder MLP backbone
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
