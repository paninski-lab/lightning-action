# Lightning Action

![GitHub](https://img.shields.io/github/license/paninski-lab/lightning-action)
![PyPI](https://img.shields.io/pypi/v/lightning-action)

A modern action segmentation framework built with PyTorch Lightning for behavioral analysis.

## Features

- **Modern Architecture**: Built with PyTorch Lightning for scalable and reproducible training
- **Multiple Backbones**: Support for TemporalMLP, RNN (LSTM/GRU), and Dilated TCN architectures
- **Command-line Interface**: Easy-to-use CLI for training and inference
- **Comprehensive Logging**: Built-in metrics tracking and visualization with TensorBoard
- **Extensive Testing**: Full test coverage for reliable development

## Installation

### Prerequisites

- Python 3.10+ 
- PyTorch with CUDA support (optional, for GPU training)

### Install from Source

```bash
git clone https://github.com/paninski-lab/lightning-action.git
cd lightning-action
pip install -e .
```

### Dependencies

Core dependencies include:
- `pytorch-lightning` - Training framework
- `torch` - Deep learning backend
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `tensorboard` - Experiment tracking

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
data/
├── markers/
│   ├── experiment1.csv
│   ├── experiment2.csv
│   └── ...
├── labels/
│   ├── experiment1.csv
│   ├── experiment2.csv
│   └── ...
└── features/  # optional, hand-crafted featurization of markers or other video representations
    ├── experiment1.csv
    ├── experiment2.csv
    └── ...
```

### 2. Create a Configuration File

Create a YAML configuration file (see `configs/segmenter_example.yaml`):

```yaml
data:
  data_path: /path/to/your/data
  input_dir: markers
  transforms:  # optional, defaults to ZScore
    - ZScore

model:
  input_size: 10
  output_size: 4
  backbone: temporalmlp
  num_hid_units: 256
  num_layers: 2
  
optimizer:
  type: Adam
  lr: 1e-3
  
training:
  num_epochs: 100
  batch_size: 32
  device: cpu  # or 'gpu'
```

### 3. Train a Model

#### Using the CLI:
```bash
litaction train --config configs/my_config.yaml --output-dir runs/my_experiment
```

#### Using the Python API:
```python
from lightning_action.api import Model

# Load model from config
model = Model.from_config('configs/my_config.yaml')

# Train model
model.train(output_dir='runs/my_experiment')
```

### 4. Generate Predictions

#### Using the CLI:
```bash
litaction predict --model-dir runs/my_experiment --data-dir /path/to/data --input-dir markers --output-dir predictions/
```

#### Using the Python API:
```python
# Load trained model
model = Model.from_dir('runs/my_experiment')

# Generate predictions
model.predict(
    data_path='/path/to/data',
    input_dir='markers',
    output_dir='predictions/'
)
```

See `configs/README.md` for detailed configuration options.

## Monitoring Training with TensorBoard

Lightning Action automatically logs training metrics to TensorBoard. To visualize your training progress:

1. **Launch TensorBoard** after starting training:
   ```bash
   tensorboard --logdir /path/to/your/runs/directory
   ```

2. **Set the correct logdir**: Use the deepest directory that contains all your model directories. For example:
   ```bash
   # If your models are in:
   # runs/experiment1/
   # runs/experiment2/
   # runs/baseline/
   
   # Launch TensorBoard with:
   tensorboard --logdir runs/
   ```

3. **Open your browser** and navigate to `http://localhost:6006` to view the TensorBoard dashboard.

4. **Available metrics** include:
   - Training and validation loss
   - Training and validation accuracy
   - Training and validation F1 score
   - Learning rate schedules

**Tip**: Keep TensorBoard running while training multiple experiments to compare results in real-time.

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=lightning_action
```

### Code Style

The project uses:
- `flake8` for linting
- `isort` for import sorting
- Maximum line length: 99 characters

## Project Structure

```
lightning_action/
├── api/           # High-level API for model usage
├── cli/           # Command-line interface
├── data/          # Data loading and preprocessing
├── models/        # Model architectures
│   └── backbones/ # Backbone implementations
└── tests/         # Test suite
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{lightning_action,
  title = {Lightning Action: A PyTorch Lightning Framework for Action Segmentation},
  author = {Whiteway, Matt},
  url = {https://github.com/paninski-lab/lightning-action},
  year = {2024}
}
```

## Acknowledgments

This framework is built upon the work of:
- [PyTorch Lightning](https://lightning.ai/) for the training framework
- [PyTorch](https://pytorch.org/) for the deep learning backend
- Previous action segmentation work from the [Paninski Lab](https://github.com/themattinthehatt/daart)
