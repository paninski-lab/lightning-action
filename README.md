# Lightning Action

![GitHub](https://img.shields.io/github/license/paninski-lab/lightning-action)
![PyPI](https://img.shields.io/pypi/v/lightning-action)
[![codecov](https://codecov.io/gh/paninski-lab/lightning-action/branch/main/graph/badge.svg)](https://codecov.io/gh/paninski-lab/lightning-action)

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
- PyTorch with CUDA support (for GPU training; optional for keypoint models, required for video models)

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

> The instructions below are for **keypoint-based models**.
> For video-based models, see [docs/video_pipeline_quickstart.md](docs/video_pipeline_quickstart.md).

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

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment,
code style, and submitting pull requests.

### Citation

If you use this framework in your research, please cite:

```bibtex
@article{blau2024study,
  title={A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised learning paradigms},
  author={Blau, Ari and Schaffer, Evan S and Mishra, Neeli and Miska, Nathaniel J and Laboratory, International Brain and Paninski, Liam and Whiteway, Matthew R},
  journal={Neurons, behavior, data analysis, and theory},
  volume={2024},
  pages={10--51628},
  year={2024}
}
```

### Funding

We are grateful for support from the following:
* Gatsby Charitable Foundation GAT3708
* [NIH R50NS145433](https://reporter.nih.gov/search/Hmj4KMmLv0evcYPlPEDa-Q/project-details/11240675)
* [NIH U19NS123716](https://reporter.nih.gov/search/Hmj4KMmLv0evcYPlPEDa-Q/project-details/11141703)
* [NSF 1707398](https://ui.adsabs.harvard.edu/abs/2017nsf....1707398A/abstract)
* [The NSF AI Institute for Artificial and Natural Intelligence](https://ui.adsabs.harvard.edu/abs/2023nsf....2229929Z/abstract)
* Simons Foundation
* Wellcome Trust 216324
* Zuckerman Institute (Columbia University) Team Science
