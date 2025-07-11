# Configuration Files

This directory contains YAML configuration files for training and evaluating action segmentation models.

## Configuration Structure

### Data Configuration
- `data.data_path`: Dataset path
- `data.sequence_length`: Length of input sequences
- `data.batch_size`: Training batch size
- `data.train_probability`: Fraction of data for training

### Model Configuration
- `model.backbone`: Backbone architecture
- `model.num_hid_units`: Backbone hidden layer size
- `model.num_layers`: Number of backbone layers

### Optimizer Configuration
- `optimizer.type`: Optimizer type ('Adam', 'AdamW')
- `optimizer.lr`: Learning rate
- `optimizer.scheduler`: Learning rate scheduler ('step', 'cosine')

### Training Configuration
- `training.max_epochs`: Maximum training epochs

## Example Configs

- `segmenter_example.yaml`: Basic segmentation model configuration
- More configs will be added for different model types and datasets

## Usage

Load and use configs in training scripts:

```python
import lightning as pl
import yaml
from lightning_action.models import Segmenter
from lightning_action.data import DataModule

# Load config
with open('configs/segmenter_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model and data module
model = Segmenter(config)
datamodule = DataModule(**config['data'])

# Train with Lightning
trainer = pl.Trainer(**config['training'])
trainer.fit(model, datamodule)
```