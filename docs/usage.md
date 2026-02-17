# Usage Guide

## Installation

```bash
# Clone and enter the repo
git clone <repo-url>
cd inertial-nav

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch matplotlib numpy scipy termcolor navpy hydra-core omegaconf wandb torchdiffeq
```

## Data Setup

### Option A: Download Pre-processed Data

Download the reformatted pickle data from the [data.zip link](https://github.com/user-attachments/files/17930695/data.zip) and extract into `data/`:

```bash
wget "https://github.com/user-attachments/files/17930695/data.zip"
unzip data.zip -d .
rm data.zip
```

This gives you pickle files for KITTI odometry sequences 00-11.

### Option B: Process from KITTI Raw

If you have KITTI raw data downloaded locally:

1. Update the raw data path in `configs/paths/local.yaml`:
   ```yaml
   data_base: "/path/to/your/KITTI/raw"
   ```

2. Run the preprocessing script:
   ```bash
   python scripts/prepare_kitti_data.py \
     --kitti_raw /path/to/KITTI/raw \
     --output data/
   ```

### Download Pretrained Weights

Download the pretrained network parameters:

```bash
wget "https://www.dropbox.com/s/77kq4s7ziyvsrmi/temp.zip"
unzip temp.zip -d temp
rm temp.zip
```

---

## Training

All training is configured through YAML files in `configs/` and driven by [Hydra](https://hydra.cc/). Any config value can be overridden from the command line.

### Quick Start

```bash
# Default training (400 epochs, RPE loss, Adam optimizer)
python src/train.py

# Fast sanity check (1 batch, 1 epoch)
python src/train.py training.debug.fast_dev_run=true
```

### Common Overrides

```bash
# Change number of epochs
python src/train.py training.epochs=100

# Disable WandB logging
python src/train.py logging.use_wandb=false

# Use a different model architecture
python src/train.py model=iekf_node_conv_cov

# Change learning rate for a specific component
python src/train.py optimizer.param_groups.init_process_cov_net.lr=1e-5

# Use ATE loss instead of RPE
python src/train.py loss=ate

# Change training subsequence length (default 6000 samples = 60s)
python src/train.py training.seq_dim=3000

# Combine overrides
python src/train.py training.epochs=50 model=iekf_node_lstm_cov logging.use_wandb=false
```

### Resume Training

```bash
python src/train.py \
  training.resume.enabled=true \
  training.resume.checkpoint_path=checkpoints/checkpoint_epoch50.pth
```

### Learning Rate Scheduling

Enable a scheduler by editing `configs/optimizer/adam.yaml` or via CLI:

```bash
# ReduceLROnPlateau
python src/train.py optimizer.scheduler.enabled=true optimizer.scheduler.type=ReduceLROnPlateau

# Cosine annealing
python src/train.py optimizer.scheduler.enabled=true optimizer.scheduler.type=CosineAnnealingLR optimizer.scheduler.T_max=100
```

### WandB Experiment Tracking

```bash
# First time: log in
wandb login

# Train with tracking enabled (default)
python src/train.py experiment.name="my-experiment" experiment.tags=["baseline"]
```

Logged metrics include `train/loss_epoch`, `train/grad_norm`, and per-group learning rates. Checkpoints are saved to the `checkpoints/` directory.

### Hyperparameter Sweeps

Hydra's multirun mode lets you sweep over parameters:

```bash
python src/train.py -m \
  optimizer.param_groups.init_process_cov_net.lr=1e-3,1e-4,1e-5 \
  training.epochs=50
```

---

## Evaluation

### Run Evaluation on Test Sequences

```bash
python src/test.py checkpoint=checkpoints/best.pth
```

This runs the IEKF filter on all test sequences, prints RPE/ATE/orientation error metrics, saves trajectory plots to `results/`, and optionally logs to WandB.

```bash
# With WandB logging
python src/test.py checkpoint=checkpoints/best.pth logging.use_wandb=true

# Without WandB
python src/test.py checkpoint=checkpoints/best.pth logging.use_wandb=false
```

### Evaluate a Legacy Checkpoint

Convert and evaluate a checkpoint from the original codebase:

```bash
python scripts/convert_legacy_checkpoint.py \
  --input temp/iekfnets.p \
  --output checkpoints/converted.pth

python src/test.py checkpoint=checkpoints/converted.pth logging.use_wandb=false
```

### Metrics

The evaluation reports three metric families:

- **RPE (Relative Pose Error)**: Translational error as a percentage of distance traveled, computed at distance windows of 100-800m. This is the primary KITTI odometry benchmark metric. The published baseline achieves **1.10%** mean RPE.
- **ATE (Absolute Trajectory Error)**: Position error in meters after optional Umeyama alignment. Reports mean, RMSE, median, and max.
- **Orientation Error**: Rotation angle error in degrees between predicted and ground truth orientations.

---

## Model Architectures

The system is modular: the IEKF filter is fixed, but the neural networks that predict its noise parameters can be swapped via config. Select a model config with `model=<name>`:

| Config | Measurement Cov Net | Notes |
|--------|-------------------|-------|
| `iekf_learned_cov` | MeasurementCovNet (CNN) | Default. Matches the original paper. |
| `iekf_node_conv_cov` | NeuralODEConvCovNet | Neural ODE with conv backbone (~3.5K params) |
| `iekf_node_lstm_cov` | NeuralODELSTMCovNet | Neural ODE with LSTM backbone (~4K params) |
| `iekf_learned_dynamics` | MeasurementCovNet + NeuralODEDynamics | Also replaces classical inertial kinematics with a learned model |

All models also include an **InitProcessCovNet** that learns initial state and process noise covariances.

### Adding a New Architecture

1. Create a new network class in `src/models/` that extends `BaseCovarianceNet`:

   ```python
   from src.models.base_covariance_net import BaseCovarianceNet

   class MyNewNet(BaseCovarianceNet):
       def __init__(self, input_channels=6, output_dim=2, **kwargs):
           super().__init__()
           # ... define layers ...

       def forward(self, u, iekf):
           # u: (N, input_channels) normalized IMU
           # return: (N, output_dim) covariance scaling factors
           ...

       def get_output_dim(self):
           return self.output_dim
   ```

2. Register it in `src/models/__init__.py`:

   ```python
   from src.models.my_new_net import MyNewNet
   register_model(MyNewNet)
   ```

3. Create a config file `configs/model/iekf_my_new.yaml`:

   ```yaml
   name: "iekf_my_new"
   _target_: configs.model.iekf_learned_cov

   networks:
     init_process_cov:
       enabled: true
       type: "InitProcessCovNet"
       architecture:
         output_dim: 6
         initial_beta: 3.0
         weight_scale: 10.0

     measurement_cov:
       enabled: true
       type: "MyNewNet"
       architecture:
         input_channels: 6
         output_dim: 2
         # ... your custom params ...
   ```

4. Train with it:

   ```bash
   python src/train.py model=iekf_my_new
   ```

---

## Configuration Reference

All config files live in `configs/`. Hydra composes them into a single config at runtime.

```
configs/
  config.yaml              # Top-level: selects defaults, experiment metadata, logging
  dataset/
    kitti.yaml             # Sequence definitions, train/val/test splits, noise params
  model/
    iekf_learned_cov.yaml  # Default model: physics params, network architectures
    iekf_node_conv_cov.yaml
    iekf_node_lstm_cov.yaml
    iekf_learned_dynamics.yaml
  training/
    default.yaml           # Epochs, seq_dim, gradient clipping, checkpointing, seed
  optimizer/
    adam.yaml              # Optimizer type, per-component learning rates, scheduler
  loss/
    rpe.yaml               # RPE loss: distance windows, criterion, normalization
    ate.yaml               # ATE loss: alignment, reduction
  paths/
    local.yaml             # Data paths, checkpoint dir, results dir
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 400 | Number of training epochs |
| `training.seq_dim` | 6000 | Training subsequence length (samples at 100Hz) |
| `training.gradient_clipping.max_norm` | 1.0 | Max gradient norm |
| `training.seed` | 42 | Random seed |
| `optimizer.type` | Adam | Optimizer (Adam, AdamW, SGD) |
| `optimizer.param_groups.*.lr` | 1e-4 | Per-component learning rates |
| `loss.type` | RPELoss | Loss function (RPELoss, ATELoss) |
| `logging.use_wandb` | true | Enable WandB tracking |
| `model.networks.*.enabled` | true | Enable/disable individual networks |

---

## Project Structure

```
src/
  core/               # IEKF filter implementation
    torch_iekf.py      #   PyTorch IEKF (training + inference)
  models/              # Neural network architectures
    base_covariance_net.py
    init_process_cov_net.py
    measurement_cov_net.py
    neural_ode_cov_net.py
    neural_ode_dynamics.py
  data/                # Data loading and transforms
    base_dataset.py
    kitti_dataset.py
    transforms.py
  losses/              # Loss functions
    rpe_loss.py
    ate_loss.py
  training/            # Training infrastructure
    trainer.py         #   Main training loop
    optimizer_factory.py
    callbacks.py       #   Checkpointing, WandB, early stopping
  evaluation/          # Metrics and visualization
    metrics.py         #   RPE, ATE, orientation error
    visualization.py   #   Trajectory and error plots
  utils/               # Shared utilities
    geometry.py        #   SO(3)/SE(3) operations
    wandb_utils.py
  train.py             # Training entry point
  test.py              # Evaluation entry point
  legacy/              # Original codebase (preserved for reference)
```

### How It Fits Together

1. **`src/train.py`** reads the Hydra config and passes it to `Trainer`.
2. **`Trainer`** builds a `TorchIEKF` model (with neural networks attached via the model registry), a loss function, an optimizer, and callbacks.
3. Each training step: load a sequence from the dataset, sample a random subsequence, add IMU noise, run the neural networks to predict covariances, run the IEKF filter, compute RPE loss, backpropagate.
4. **`src/test.py`** loads a trained checkpoint, runs `TorchIEKF` with `torch.no_grad()` to get covariances and filter output, and computes evaluation metrics.

---

## Running Tests

```bash
# All tests (173 tests, no KITTI data needed)
python -m pytest tests/ -v

# By component
python -m pytest tests/test_geometry.py -v       # SO3/SE3 math
python -m pytest tests/test_torch_iekf.py -v     # PyTorch filter
python -m pytest tests/test_models.py -v         # Neural networks
python -m pytest tests/test_datasets.py -v       # Data pipeline
python -m pytest tests/test_losses.py -v         # Loss functions
python -m pytest tests/test_trainer.py -v        # Training loop
python -m pytest tests/test_evaluation.py -v     # Metrics & plots
```

---

## Legacy Code

The original implementation is preserved in `src/legacy/`. To run it directly:

```bash
cd src/legacy
python main_kitti.py
```

See `src/legacy/README.md` for details. Use `scripts/convert_legacy_checkpoint.py` to convert old checkpoints to the modern format.
