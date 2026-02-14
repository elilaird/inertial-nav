# LIO-V: Learned Inertial Odometry for Vehicles — Design Document

## Overview

A deep learning model that acts as a "Virtual IMU" — takes noisy, raw IMU sensor data and outputs clean trajectories by learning the latent constraints of vehicle dynamics (a "World Model"). Based on the TLIO architecture adapted for vehicles.

## Decisions

- **Architecture:** Pure 1D ResNet-18 (Option A). Can extend to ResNet+LSTM later.
- **Datasets:** KITTI Odometry (primary) + nuScenes (secondary).
- **Environment:** PyTorch + Conda. Develop/test on Apple Silicon (MPS), full training on HPC (CUDA).
- **Experiment tracking:** Weights & Biases.
- **Visualization:** Static matplotlib plots (PDF + PNG).
- **Sample rates:** Native per dataset (KITTI 10Hz, nuScenes ~50Hz), variable window sizes.

## Project Structure

```
inertial-nav/
├── configs/                  # YAML experiment configs
│   ├── default.yaml
│   ├── kitti.yaml
│   └── nuscenes.yaml
├── data/                     # Data directory (gitignored)
│   ├── raw/                  # Downloaded datasets
│   └── processed/            # Windowed, preprocessed data
├── src/
│   ├── data/
│   │   ├── kitti_loader.py       # KITTI OxTS parser
│   │   ├── nuscenes_loader.py    # nuScenes IMU parser
│   │   ├── noise_augmentation.py # Synthetic degradation
│   │   ├── windowing.py          # Window creation + ground truth
│   │   └── dataset.py            # PyTorch Dataset class
│   ├── models/
│   │   ├── resnet1d.py           # 1D ResNet-18
│   │   └── heads.py              # Output heads (displacement + heading)
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # MSE + consistency loss
│   │   └── metrics.py            # Drift %, ATE, RTE
│   ├── inference/
│   │   ├── dead_reckoning.py     # Trajectory reconstruction loop
│   │   └── evaluate.py           # Full pipeline evaluation
│   └── visualization/
│       ├── trajectory_plot.py    # The "red vs green" money shot
│       └── metrics_plot.py       # Error over distance plots
├── scripts/
│   ├── download_kitti.sh
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   ├── test_noise.py
│   └── test_dead_reckoning.py
├── environment.yml               # Conda environment
└── docs/plans/
```

## Data Pipeline

### KITTI Loader
- Parse OxTS data files (lat/lon/alt, roll/pitch/yaw, velocities, accelerations, angular rates) from sequences 00-10
- Extract 6-channel IMU: 3-axis accelerometer + 3-axis gyroscope at 10Hz (native rate)
- Compute ground truth poses from GPS/INS solution

### nuScenes Loader
- Parse IMU data from CANbus expansion (accelerometer + gyroscope) at ~50Hz
- Ground truth from ego_pose (lidar-based localization)

### Noise Augmentation (Synthetic Degradation)
- **White noise:** Gaussian on accel (sigma=0.5 m/s^2) and gyro (sigma=0.01 rad/s), configurable
- **Bias drift:** Random walk bias accumulating over time, simulating cheap MEMS sensors
- **Noise levels as config params** for sweep experiments

### Windowing
- 1-second windows at native rate (10 samples for KITTI, ~50 for nuScenes)
- Ground truth label per window: relative displacement (dx, dy, dz) and heading change (dpsi) in body frame
- 50% overlap between consecutive windows
- Train/val/test split by sequence (KITTI: 00-06 train, 07-08 val, 09-10 test)

## Model Architecture

### 1D ResNet-18
- Input: `(batch, 6, window_len)` — 6 IMU channels, variable window length
- 2D convolutions replaced by 1D convolutions
- 4 residual blocks: [64, 128, 256, 512] channels
- Global average pooling → 512-dim feature vector
- Adaptive pooling handles variable window lengths across datasets

### Output Head
- FC layers: 512 -> 128 -> 4
- Output: `(dx, dy, dz, dpsi)` — 3D displacement + heading change in body frame
- No activation (regression)

### Losses
- **Primary:** MSE between predicted and ground truth `(dx, dy, dz, dpsi)`
- **Consistency loss (optional, config toggle):** Overlapping window displacement sums should match combined span. Weighted by lambda.

### Metrics
- **ATE:** RMSE of position after full dead-reckoning reconstruction
- **RTE:** Average drift as % error over fixed distances (100m, 200m)
- **Per-axis error:** Verify non-holonomic constraint is learned (lateral error near zero)

## Training & Experiments

### Training Setup
- Adam optimizer, lr=1e-3, ReduceLROnPlateau scheduler
- Batch size 128 (adjustable for MPS vs HPC)
- Early stopping on validation loss (patience=10)
- Best model checkpointing by val loss
- W&B logging: loss curves, lr, per-axis errors, sample trajectory reconstructions

### Experiments
1. **Baseline:** Raw noisy integration (no model) — the "red line"
2. **KITTI default noise:** Train seq 00-06, val 07-08, test 09-10
3. **Noise sweep:** Vary noise sigma to show robustness across sensor quality
4. **nuScenes transfer:** Train on KITTI, evaluate on nuScenes (zero-shot)
5. **nuScenes fine-tuned:** Fine-tune KITTI model on nuScenes training split

## Visualization

### Trajectory Plot ("Money Shot")
- Black line: Ground truth
- Red dotted line: Raw noisy integration
- Green line: Model output
- One plot per test sequence

### Supporting Plots
- Error vs. distance traveled (drift rate)
- Per-axis displacement error distribution (lateral should cluster near zero)
- Noise level vs. ATE (from sweep experiment)

All plots saved as PDF + PNG, logged as W&B artifacts.

## Tests
- **test_data_pipeline.py:** Window shapes, noise augmentation stats, ground truth label correctness
- **test_model.py:** Forward pass shape checks, gradient flow, variable window length handling
- **test_noise.py:** Noise distribution matches expected parameters
- **test_dead_reckoning.py:** Given perfect predictions, reconstructed trajectory matches ground truth exactly

## Device Strategy
- `get_device()` utility: auto-detect CUDA / MPS / CPU
- `--no-wandb` flag for local dev without W&B
- Configs via YAML with CLI overrides (argparse + yaml)
