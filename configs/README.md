# Configuration Files

This directory contains YAML configuration files for training and evaluating action segmentation models.

## Configuration Structure

### Data Configuration
- `data.data_path`: Dataset path
- `data.sequence_length`: Length of input sequences
- `data.batch_size`: Training batch size
- `data.train_probability`: Fraction of data for training
- `data.transforms`: List of transforms for input data (optional, default is ZScore) 

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
from lightning_action.api import Model

# Load model from config
model = Model.from_config('configs/segmenter_example.yaml')

# Train model
model.train(output_dir='/path/to/output')
```
