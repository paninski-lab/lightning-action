"""Video segmentation models with Lightning integration.

This module contains the video segmentation model that processes 
chunks of video frames through a pretrained ViT-MAE backbone 
(optionally frozen), pools patch embeddings, appends differencing 
for temporal features, passes through a flexible temporal backbone 
(e.g., DTCN, RNN, or TemporalMLP), and classifies per-frame behaviors. 
"""

import logging
from abc import abstractmethod
from typing import Any, List, Tuple, Union

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from jaxtyping import Float, Int
from torchmetrics import Accuracy, F1Score
from typeguard import typechecked
from transformers import ViTMAEModel
from lightning.pytorch.utilities import rank_zero_only

from lightning_action.models.backbones import DilatedTCN, TemporalMLP, RNN

logger = logging.getLogger(__name__)

class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling layer for aggregating patch embeddings."""

    @typechecked
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # Initialize query parameter
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Initialize multihead attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    @typechecked
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat query for batch size
        query = self.query.repeat(x.size(0), 1, 1)
        out, _ = self.attn(query, x, x)
        return out.squeeze(1)

class VideoBaseModel(pl.LightningModule):
    """Base Lightning model for video action segmentation.
    
    Uses integer labels for efficiency.
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
        self.sequence_length = self.model_config.get('sequence_length', 128)

        # ignore index
        self.ignore_index = config.get('data', {}).get('ignore_index', -100)

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

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: Int[torch.Tensor, 'batch sequence'],
        stage: str = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and metrics.
        
        Args:
            outputs: model outputs dictionary
            targets: ground truth labels (integer class indices)
            stage: training stage ('train', 'val', 'test')
            
        Returns:
            tuple of (loss tensor, metrics dictionary)
        """
        logits = outputs['logits']
        if hasattr(self, 'num_lags') and self.num_lags > 0:
            logits = logits[:, self.num_lags:-self.num_lags, :]
            probabilities = F.softmax(logits, dim=-1)
        else:
            probabilities = outputs['probabilities']

        # flatten for loss computation
        logits_flat = logits.view(-1, self.output_size)
        targets_flat = targets.view(-1) 

        # Get class weights from config and move to the correct device
        class_weights = self.model_config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=self.device, dtype=torch.float)
            
        # compute cross entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            weight=class_weights,
        )

        # compute metrics
        with torch.no_grad():
            probs_flat = probabilities.view(-1, self.output_size)
            pred_classes = torch.argmax(probs_flat.clone(), axis=-1)

            if stage == 'train':
                accuracy = self.train_accuracy(pred_classes, targets_flat)
                f1 = self.train_f1(pred_classes, targets_flat)
            else:  # val or test
                accuracy = self.val_accuracy(pred_classes, targets_flat)
                f1 = self.val_f1(pred_classes, targets_flat)

        # handle NaN losses
        loss_value = loss.item()
        accuracy_value = accuracy.item()
        f1_value = f1.item()
        
        metrics = {}
        if not torch.isnan(loss):
            metrics[f'{stage}_loss'] = loss_value
        if not torch.isnan(accuracy):
            metrics[f'{stage}_accuracy'] = accuracy_value
        if not torch.isnan(f1):
            metrics[f'{stage}_f1'] = f1_value
        
        return loss, metrics

    @typechecked
    def training_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step for the model.
        
        Args:
            batch: tuple or list of (frames, labels)
            batch_idx: index of the batch
            
        Returns:
            loss tensor
        """
        frames, labels = batch
        outputs = self.forward(frames)
        loss, metrics = self.compute_loss(outputs, labels, stage='train')
        
        # log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', metrics['train_accuracy'], prog_bar=True)
        self.log('train_f1', metrics['train_f1'], prog_bar=True)
        
        return loss

    @typechecked
    def validation_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
        batch_idx: int
    ) -> None:
        """Validation step for the model.
        
        Args:
            batch: tuple or list of (frames, labels)
            batch_idx: index of the batch
        """
        frames, labels = batch
        outputs = self.forward(frames)
        loss, metrics = self.compute_loss(outputs, labels, stage='val')
        
        # log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', metrics['val_accuracy'], prog_bar=True)
        self.log('val_f1', metrics['val_f1'], prog_bar=True)

    @typechecked
    def predict_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Prediction step for the model.
        
        Args:
            batch: tuple or list of (frames, labels)
            batch_idx: index of the batch
            dataloader_idx: index of the dataloader (optional)
            
        Returns:
            dictionary with predictions
        """
        frames, _ = batch  # ignore labels for prediction
        outputs = self.forward(frames)
        if hasattr(self, 'num_lags') and self.num_lags > 0:
            sl = slice(self.num_lags, -self.num_lags)
            return {
                'logits': outputs['logits'][:, sl, :],
                'probabilities': outputs['probabilities'][:, sl, :],
                'predictions': torch.argmax(outputs['probabilities'][:, sl, :], dim=-1),
            }
        else:
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
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(lr),
                weight_decay=float(weight_decay),
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


class VideoSegmenter(VideoBaseModel):
    """Video segmentation model for action recognition.
    
    Processes video chunks through ViT-MAE, pooling, differencing, and a flexible backbone.
    """

    @typechecked
    def __init__(self, config: dict[str, Any]):
        """Initialize VideoSegmenter.
        
        Args:
            config: configuration dictionary with model, optimizer, and training settings
        """
        self.embed_dim = 768  # ViT-MAE base embedding dim
        self.num_heads = config.get('model', {}).get('num_heads', 12)  # Default to match ViT
        self.num_lags = config.get('model', {}).get('num_lags', 0)
        super().__init__(config)
        
        # load ViT-MAE backbone
        self.beast = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
        beast_ckpt = self.model_config.get('beast_checkpoint')
        if beast_ckpt and os.path.exists(beast_ckpt):
            checkpoint = torch.load(beast_ckpt, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            for prefix in ['beast.', 'vit_mae.vit.']:
                beast_state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)}
                if beast_state_dict:
                    break
            else:
                beast_state_dict = state_dict
            self.beast.load_state_dict(beast_state_dict, strict=False)
            logger.info(f"Loaded BEAST checkpoint: {beast_ckpt}")
        
        self.freeze_beast = self.model_config.get('freeze_beast', True)
        for param in self.beast.parameters():
            param.requires_grad = not self.freeze_beast
        
        # log model initialization
        logger.info(f"VideoSegmenter initialized, freeze_beast={self.freeze_beast}")
        
        # initialize pooling
        self.pooling = MultiHeadAttentionPooling(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
        # build backbone and classifier
        self._build_model()

    def _build_model(self):
        """Build the segmentation model architecture."""
        self.backbone = self._build_backbone()
        backbone_output_size = self._get_backbone_output_size()
        self.classifier = nn.Linear(backbone_output_size, self.output_size)
        self._initialize_weights()

    def _build_backbone(self) -> nn.Module:
        """Build the backbone network.
        
        Returns:
            backbone network module
        """
        backbone_type = self.model_config.get('backbone', 'dtcn')

        logger.info(f'Constructing VideoSegmenter model with {backbone_type} backbone')
        
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
        """Get the output size of the backbone network.
        
        Returns:
            output feature size of the backbone
        """
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
        x: Float[torch.Tensor, 'batch sequence channels height width'],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the video segmentation model.
        
        Args:
            x: input tensor with shape (batch, sequence, 3, height, width)
            
        Returns:
            dictionary with 'logits', 'probabilities', 'features'
        """
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        # extract embeddings
        with torch.no_grad() if self.freeze_beast else torch.enable_grad():
            embeddings = self.beast(pixel_values=x).last_hidden_state[:, 1:, :]
        
        # pool
        pooled = self.pooling(embeddings).view(b, s, -1)
        
        # differencing
        diffs = pooled[:, 1:] - pooled[:, :-1]
        zero_diff = torch.zeros(b, 1, pooled.shape[-1], device=pooled.device)
        diffs = torch.cat([zero_diff, diffs], dim=1)
        
        # concatenate features
        features = torch.cat([pooled, diffs], dim=-1)
        
        # backbone
        backbone_output = self.backbone(features)
        
        # classifier
        logits = self.classifier(backbone_output)
        
        # probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': features,
        }
