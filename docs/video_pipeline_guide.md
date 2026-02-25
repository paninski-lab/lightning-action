# Video Pipeline Reading Guide

This guide walks you through the video action segmentation pipeline in reading order — top-down, from the highest-level entry point to the lowest-level internals. By the end, you'll understand how raw video files become frame-level action predictions.

The video pipeline sits alongside the existing CSV/marker pipeline. Both share a common base (model, training utilities, data utilities), but the video pipeline adds GPU-accelerated video decoding via NVIDIA DALI and a backbone-neck-head architecture for learning directly from pixels.

---

## Recommended Reading Order

### 1. `lightning_action/api/video_model.py` — VideoModel

**Start here.** This is the highest-level entry point and shows what the pipeline *does*: load a config, train a model, and run predictions on video files.

```python
# Train from config
model = VideoModel.from_config("config.yaml")
model.train(output_dir="runs/my_experiment")

# Load and predict
model = VideoModel.from_dir("runs/my_experiment")
model.predict(videos_dir="/path/to/videos", output_dir="predictions/")
```

Key things to notice:

- **`train()`** delegates to `train_video()` (covered in step 3), then optionally runs post-training inference on all training videos.
- **`predict()`** creates a temporary DALI pipeline per video, runs the model, and reassembles per-chunk predictions into a single frame-level prediction array. The boundary reassembly logic — slicing away overlap between consecutive chunks — is what makes this non-trivial.
- **`_get_video_frame_count()`** uses OpenCV to determine how many frames a video has, which is needed to verify that predictions cover the full video.
- Inherits from `BaseModelAPI` (next file), so config loading, checkpoint discovery, and `from_config()` / `from_dir()` factory methods are all inherited.

### 2. `lightning_action/api/base.py` — BaseModelAPI

The abstract base class shared by the CSV `Model` and `VideoModel`. Read this alongside `video_model.py` to see what's inherited versus overridden.

Provides:

- **`from_config(config_path)`** — Load a YAML config, instantiate the model, return a ready-to-train API object.
- **`from_dir(model_dir)`** — Discover a trained checkpoint, load the model from it, return a ready-to-predict API object.
- **Checkpoint discovery** — Finds the best checkpoint in the output directory (by validation loss).
- **`chdir()` context manager** — Temporarily changes the working directory during training, since relative paths in configs are resolved from the output directory.

Subclasses must implement six abstract methods: `_create_model_from_config()`, `_get_model_class()`, `_get_train_function()`, `_setup_trainer()`, `_run_post_training_inference()`, and `predict()`.

### 3. `lightning_action/video_train.py` — train_video()

The training orchestrator. This single function is the full training flow:

1. **Seed setup** — `reset_seeds()` for reproducibility (Python, NumPy, PyTorch, cuDNN).
2. **DataModule creation** — Instantiates `VideoDataModule` from the config's `data` section.
3. **Class weights** — After `datamodule.setup()`, class weights computed by `VideoDataset` are injected back into the config so the model's loss function can use them.
4. **Model instantiation** — Creates or reuses a `VideoSegmenter` with the updated config.
5. **Trainer configuration** — TensorBoardLogger, callbacks (checkpoint, early stopping, LR monitor), and DDP settings.
6. **`trainer.fit()`** — Runs training.
7. **Config saving** — Writes the final config (with computed class weights, label names, etc.) to `output_dir/config.yaml`.

Key decisions to note:

- **`mp.set_start_method("spawn")`** — Required for DALI + DDP compatibility. Fork-based multiprocessing doesn't play well with CUDA contexts.
- **`use_distributed_sampler=False`** — DALI handles its own data sharding across GPUs, so Lightning's built-in distributed sampler is disabled.
- **NCCL backend** — Used for multi-GPU gradient synchronization.

### 4. `lightning_action/data/video_dataset.py` — VideoDataset

A metadata container that discovers and validates video/label pairs. It does *not* load frames — that's DALI's job (next file).

Responsibilities:

- **File discovery** — Finds `.mp4` files in `videos_dir` and matching `.npy` label files in `labels_dir`, filtered by an `expt_ids` list from the config.
- **Validation** — Ensures every video has a corresponding label file and that frame counts match.
- **Class info extraction** — Scans all label arrays to determine the set of unique classes and their names.
- **Class weight computation** — Computes inverse-frequency weights for the loss function, handling class imbalance.
- **TCN padding** — Computes `sequence_pad` (the number of extra frames needed on each side of a chunk so the TCN has enough temporal context).

Key decision: **`.npy` label files** instead of CSV. Each `.npy` file is a 1-D integer array where `labels[i]` is the class index for frame `i`. This is much faster to load than parsing CSV files, especially for long videos.

### 5. `lightning_action/data/video_datamodule.py` — VideoDataModule, VideoPipeline, DALIIterator

The heart of the data pipeline. Three classes in one file, each handling a different layer of abstraction.

**VideoDataModule** (Lightning DataModule):
- Creates a `VideoDataset`, then splits *videos* (not chunks) into train and val sets.
- Builds a `VideoPipeline` + `DALIIterator` for each split.
- Exposes `train_dataloader()` and `val_dataloader()` for Lightning's training loop.

**VideoPipeline** (DALI Pipeline):
- Defines the GPU-accelerated preprocessing graph: video decoding, resizing, and normalization.

**DALIIterator**:
- Wraps DALI's output tensors and attaches labels and per-chunk metadata.
- Metadata includes: video index, start frame, and boundary flags (`is_first_chunk`, `is_last_chunk`). These flags are consumed by the model during prediction to correctly handle overlap at video boundaries.
- Produces batches of shape `(B, T, C, H, W)` — batch, time, channels, height, width.

Key decisions:

- **Video-level train/val splits** — If chunks from the same video appeared in both train and val, the model could memorize video-specific patterns and appear to generalize when it hasn't. Splitting at the video level prevents this data leakage.
- **DALI for video I/O** — CPU-based video decoding (e.g., OpenCV or torchvision) is the bottleneck in video training. DALI decodes on the GPU, keeping the pipeline GPU-bound rather than CPU-bound.
- **Extended sequences** — Each chunk includes extra frames (the TCN's receptive field padding) so the temporal head has enough context. The padding frames are included in the input but their predictions are discarded.

### 6. `lightning_action/models/video_segmenter.py` — VideoBaseModel + VideoSegmenter

The model itself, split into two classes.

**VideoBaseModel** (inherits `BaseModel`):
- Overrides `_get_inputs_and_targets()` to unpack DALI's tuple batches `(frames, labels, metadata)` instead of the CSV pipeline's `(features, labels)`.
- Adds **boundary-aware prediction slicing** — during validation and prediction, the first and last chunks in a video are handled differently to avoid double-counting frames or missing frames at boundaries.
- Implements **DDP-safe batch skipping** — when a batch contains only ignored frames (e.g., all padding), all GPUs must still participate in the forward pass to avoid NCCL hangs.

**VideoSegmenter** (inherits `VideoBaseModel`):
- Builds the full architecture in `_build_model()`:
  1. **Backbone** — one of `ResNetBackbone`, `ResNetBeastBackbone`, or `ViTMAEBackbone` (config key: `backbone`)
  2. **Attention pooling neck** — `MultiheadAttentionPooling` collapses spatial features into frame-level vectors
  3. **Velocity concat** — doubles the feature dimension by appending frame-to-frame differences
  4. **Temporal head** — `DilatedTCN`, `TemporalMLP`, or `RNN` (config key: `head`)
  5. **Linear classifier** — maps head output to class logits
- `forward()` processes frames through this chain: reshape `(B, T, C, H, W)` to `(B*T, C, H, W)`, encode with backbone, pool, reshape back to `(B, T, D)`, run temporal head, classify.

Key decisions:

- **Separate optimizer parameter groups** — The backbone is pretrained and gets a separate learning rate (or is fully frozen via `freeze_backbone: true`). The head and classifier, initialized from scratch, get a different learning rate. This prevents the pretrained features from being destroyed early in training.
- **Auto-computed `input_size`** — The temporal head's input size is determined automatically from the backbone's `hidden_size` (and doubled if velocity concat is enabled), so the user doesn't need to manually compute feature dimensions.

### 7. `lightning_action/models/backbones/` — Frame Encoders

Three interchangeable backbone architectures that share a common interface:

| Property / Method | Purpose |
|---|---|
| `hidden_size` | Output feature dimension (e.g., 512 for ResNet-18) |
| `forward(x)` | `(B, C, H, W)` images in, `(B, D, H', W')` spatial features out |
| `load_pretrained_weights(path)` | Load weights from a checkpoint file |
| `get_last_layer_params()` | Returns parameters of the final layer (for fine-tuning) |

**`resnet.py` — ResNetBackbone** (read this one first):
- Wraps a torchvision ResNet. Removes the final average pool and classification head so it outputs spatial feature maps.
- Simplest implementation; good starting point for understanding the interface.

**`resnet_beast.py` — ResNetBeastBackbone**:
- A custom ResNet implementation compatible with checkpoints from the `beast` autoencoder package.
- Contains its own `BasicBlock` and `Bottleneck` definitions rather than using torchvision's.

**`vitmae.py` — ViTMAEBackbone**:
- Wraps HuggingFace's `ViTMAEModel`. Processes frames through patch embedding + transformer encoder.
- Outputs spatial features of shape `(B, D, H', W')` where `H' = W' = image_size / patch_size`.

Key decision: **Uniform interface** — Because all three backbones expose the same properties and methods, swapping backbones is a single config change (`backbone: resnet18` vs `backbone: vitmae`). No code changes required.

### 8. `lightning_action/models/necks/mha_pooling.py` — MultiheadAttentionPooling

Bridges the gap between spatial features (a grid of patch embeddings from the backbone) and the temporal head (which expects one vector per frame).

Implements **Pooling by Multi-head Attention (PMA)** from the Set Transformer paper:
- Learnable **seed vectors** serve as queries.
- Backbone patch features (flattened from `H' x W'` grid to a sequence) are keys and values.
- Multi-head cross-attention produces one output vector per seed.
- With `num_seeds=1` (the default), this collapses a `(B, num_patches, D)` tensor to `(B, 1, D)`, giving a single vector per frame.

An optional feedforward network (FFN) with residual connection and layer normalization can be applied after the attention step.

### 9. `lightning_action/models/segmenter.py` — BaseModel

The parent class for *all* models — both CSV and video. Read this to understand what `VideoBaseModel` inherits and what it overrides.

Provides:
- **Metrics** — `torchmetrics.Accuracy` and `torchmetrics.F1Score` for train and validation.
- **Loss computation** — Cross-entropy with optional class weights and an ignore index for padding frames.
- **Optimizer configuration** — Adam or AdamW, with optional cosine annealing LR scheduler.
- **Padding removal** — Strips padding from predictions and targets before computing metrics.
- **Abstract methods** — `_build_model()` and `forward()` must be implemented by subclasses.

Also contains `Segmenter`, the concrete CSV-based model. Comparing `Segmenter` with `VideoSegmenter` shows exactly what the video pipeline adds: a backbone, a neck, and different batch handling.

### 10. Shared Utilities

Two utility modules that both the CSV and video pipelines depend on.

**`lightning_action/data/utils.py`**:
- `compute_sequence_pad(model_type, num_layers, ...)` — Calculates the number of padding frames the TCN needs for its receptive field.
- `split_sizes_from_probabilities(n, probs)` — Converts fractional split ratios (e.g., `[0.8, 0.2]`) to integer counts that sum to `n`.
- `compute_class_weights(labels)` — Inverse-frequency weighting for imbalanced class distributions.
- `ZScore`, `MotionEnergy`, `VelocityConcat`, `Compose` — Composable data transforms used by the CSV pipeline. The video pipeline handles its transforms in DALI instead.

**`lightning_action/train_utils.py`**:
- `reset_seeds(seed)` — Sets random seeds across Python, NumPy, and PyTorch for reproducibility.
- `get_callbacks(output_dir, ...)` / `get_callbacks_from_config(config, output_dir)` — Configures `ModelCheckpoint`, `EarlyStopping`, and `LearningRateMonitor` callbacks.
- `validate_config(config, required_sections)` — Asserts that required top-level config sections exist.
- `save_config(config, output_dir)` — Writes the final config to YAML, decorated with `@rank_zero_only` for DDP safety.

These were extracted from the original CSV training code so both pipelines share the same infrastructure.

---

## Key Design Decisions

### 1. DALI for Video I/O
CPU-based video decoding is the dominant bottleneck in video model training. DALI decodes video directly on the GPU, keeping the entire pipeline GPU-bound. A runtime availability check (`DALI_AVAILABLE` flag) allows the codebase to import cleanly on machines without DALI installed.

### 2. Video-Level Train/Val Splits
Splitting at the chunk level would let fragments of the same video appear in both train and val sets, creating data leakage. The pipeline splits at the video level: each video is assigned entirely to train or val.

### 3. Backbone Interface Pattern
All backbones expose `hidden_size`, `forward()`, `load_pretrained_weights()`, and `get_last_layer_params()`. This uniform interface means swapping from a ResNet to a ViT-MAE is a single config change — no code modifications needed.

### 4. Boundary-Aware Prediction
Videos are processed in overlapping chunks (because the TCN needs temporal context on each side). During prediction, the first and last chunks in a video are sliced differently to avoid double-counting frames at boundaries and to ensure every frame gets exactly one prediction.

### 5. Separate Optimizer Parameter Groups
The backbone (pretrained on ImageNet or similar) gets a lower learning rate or is fully frozen. The temporal head and classifier, initialized randomly, get the full learning rate. This protects pretrained features from being destroyed before the head learns to use them.

### 6. BaseModel Inheritance
The video model reuses the CSV pipeline's metrics, loss computation, optimizer configuration, and padding logic. `VideoBaseModel` only overrides batch handling and prediction slicing. This keeps the two pipelines consistent and reduces duplicated code.

