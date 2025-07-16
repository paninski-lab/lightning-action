# Configuration Files

This directory contains YAML configuration files for training and evaluating action segmentation models.

## Configuration Structure

### Data Configuration
- `data.data_path`: Path to dataset directory containing signal subdirectories
- `data.input_dir`: Name of input signal directory (e.g., 'markers', 'features')
- `data.expt_ids`: List of experiment IDs to use (null for all experiments)
- `data.transforms`: List of transform class names for input data preprocessing (optional, defaults to ['ZScore'])
- `data.ignore_index`: Class index to ignore in loss computation (typically 0 for background class)
- `data.weight_classes`: Whether to compute class weights for imbalanced datasets
- `data.seed`: Random seed for data splitting

### Model Configuration
- `model.input_size`: Size of input feature dimension
- `model.output_size`: Number of output classes
- `model.backbone`: Backbone architecture ('temporalmlp', 'rnn', 'dilatedtcn')
- `model.num_hid_units`: Number of hidden units in backbone layers
- `model.num_layers`: Number of layers in backbone network
- `model.num_lags`: Number of temporal lags for TemporalMLP and DilatedTCN
- `model.activation`: Activation function ('relu', 'lrelu', 'gelu')
- `model.dropout_rate`: Dropout rate for regularization
- `model.seed`: Random seed for model initialization

#### RNN-specific Parameters
- `model.rnn_type`: Type of RNN ('lstm', 'gru')
- `model.bidirectional`: Whether to use bidirectional RNN

### Optimizer Configuration
- `optimizer.type`: Optimizer type ('Adam', 'AdamW')
- `optimizer.lr`: Learning rate
- `optimizer.wd`: Weight decay coefficient
- `optimizer.scheduler`: Learning rate scheduler ('step', 'cosine', or null)

#### Scheduler Parameters
For step scheduler:
- `optimizer.step_size`: Step size for learning rate decay
- `optimizer.gamma`: Multiplicative factor for learning rate decay

For cosine scheduler:
- `optimizer.T_max`: Maximum number of iterations for cosine annealing

### Training Configuration
- `training.device`: Device to use for training ('cpu' or 'gpu')
- `training.num_epochs`: Maximum number of training epochs
- `training.min_epochs`: Minimum number of training epochs
- `training.batch_size`: Training batch size
- `training.sequence_length`: Length of input sequences
- `training.sequence_pad`: Padding for sequences
- `training.num_workers`: Number of data loading workers
- `training.train_probability`: Fraction of data for training
- `training.val_probability`: Fraction of data for validation
- `training.ckpt_every_n_epochs`: Frequency of periodic checkpointing

## Available Transforms

The framework supports the following data transforms:

- **ZScore**: Z-score normalization (mean=0, std=1)
- **MotionEnergy**: Compute motion energy (absolute differences)
- **VelocityConcat**: Compute velocity and concatenate with original signal
- **Compose**: Chain multiple transforms together

### Transform Configuration

```yaml
data:
  transforms:  # Single transform
    - ZScore  
  # or
  transforms:  # Multiple transforms applied sequentially
    - ZScore
    - MotionEnergy
```

## Supported Backbone Architectures

### TemporalMLP
Multi-layer perceptron with temporal convolutions for sequence modeling.

Required parameters:
- `num_hid_units`: Hidden layer size
- `num_layers`: Number of layers
- `num_lags`: Number of temporal lags

### RNN
Recurrent neural network with LSTM or GRU cells.

Required parameters:
- `num_hid_units`: Hidden state size
- `num_layers`: Number of RNN layers
- `rnn_type`: 'lstm' or 'gru'
- `bidirectional`: true/false

### DilatedTCN
Dilated temporal convolutional network for sequence modeling.

Required parameters:
- `num_hid_units`: Number of channels
- `num_layers`: Number of dilated layers
- `num_lags`: Kernel size for convolutions


## Data Directory Structure

The framework expects data organized as follows:

```
data_path/
├── markers/           # Input signal directory
│   ├── experiment1.csv
│   ├── experiment2.csv
│   └── ...
├── labels/            # Ground truth labels
│   ├── experiment1.csv
│   ├── experiment2.csv
│   └── ...
└── features_0/        # Optional additional features
    ├── experiment1.csv
    ├── experiment2.csv
    └── ...
```

## Usage

### Loading and Training with Config

```python
from lightning_action.api import Model

# Load model from config
model = Model.from_config('configs/my_config.yaml')

# Train model
model.train(output_dir='runs/my_experiment')
```

### CLI Usage

```bash
# Train with config file
litaction train --config configs/my_config.yaml --output-dir runs/my_experiment

# Predict with trained model
litaction predict --model-dir runs/my_experiment --data-path /path/to/data --input-dir markers --output-dir predictions/
```

## Configuration Validation

The framework performs validation on configuration files to ensure:
- Required parameters are present
- Parameter types are correct
- Backbone-specific parameters are valid
- Transform names are recognized
- File paths exist (for data_path)

Invalid configurations will raise descriptive error messages during model initialization.