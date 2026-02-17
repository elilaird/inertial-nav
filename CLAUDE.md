# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-IMU Dead-Reckoning system implementing accurate vehicle localization using only Inertial Measurement Unit (IMU) data. The approach combines an Invariant Extended Kalman Filter (IEKF) with deep neural networks that dynamically adapt noise parameters in real-time. Achieves 1.10% translational error on KITTI odometry sequences.

## Core Architecture

The system consists of two main blocks:

1. **IEKF Filter Block** (`utils_numpy_filter.py`)
   - Integrates inertial measurements (gyroscope and accelerometer)
   - Exploits zero lateral and vertical velocity constraints
   - Estimates 3D position, velocity, orientation, and IMU biases
   - NumPy-based implementation for runtime inference

2. **Neural Network Adapter Block** (`utils_torch_filter.py`)
   - `InitProcessCovNet`: Learns initial state and process noise covariances
   - `MesNet`: CNN that converts raw IMU signals into measurement covariance matrices
   - PyTorch-based for training and forward pass during testing
   - No knowledge of state estimates required - operates directly on IMU signals

The dual implementation approach (NumPy + PyTorch) allows the trained neural networks to predict covariances which are then used by the fast NumPy filter during testing.

## Key Files

- `main_kitti.py` - Main entry point defining the KITTI dataset class, parameters, and workflow control
- `dataset.py` - Base dataset class handling data loading, normalization, and noise injection
- `utils_torch_filter.py` - PyTorch IEKF with neural network components (training & inference)
- `utils_numpy_filter.py` - Pure NumPy IEKF implementation (fast inference)
- `train_torch_filter.py` - Training loop using relative pose error (RPE) loss
- `utils_plot.py` - Results visualization and error metrics
- `utils.py` - Data preparation and Umeyama alignment utilities

## Common Development Commands

### Testing (Default Mode)
```bash
cd src
python3 main_kitti.py
```
This runs the filter on all KITTI sequences and plots results. Sequence 02 is the true test set (sequences 00, 01, 04-11 are used for training).

### Training
Edit `main_kitti.py` line 473:
```python
train_filter = 1  # Change from 0 to 1
```
Then run:
```bash
cd src
python3 main_kitti.py
```

### Data Preparation
First download KITTI raw data, then edit `main_kitti.py` line 472:
```python
read_data = 1  # Change from 0 to 1
```
This converts KITTI raw data to pickle format in the `data/` directory.

### Running a Specific Sequence
Modify the loop in `test_filter()` function (line 431 in `main_kitti.py`) to filter by dataset name or index.

## Data Flow

1. **Input**: KITTI raw IMU data (gyro, accelerometer) + ground truth poses
2. **Preprocessing**: Convert to pickle format, normalize IMU signals, compute ground truth relative poses
3. **Training**: Neural networks learn to predict measurement covariances by minimizing RPE loss
4. **Testing**: Neural nets predict covariances → NumPy IEKF runs filter → Results saved to `results/`
5. **Visualization**: Plot trajectories, compute error metrics (RPE, ATE)

## Important Dataset Details

- **Training sequences**: All from `datasets_train_filter` dictionary in KITTIDataset (lines 122-129)
- **Validation**: `2011_09_30_drive_0028_extract` (partial sequence)
- **Test sequence**: `2011_09_30_drive_0028_extract` (seq 02 in KITTI odometry benchmark)
- **Sampling rate**: 100 Hz (10ms intervals)
- **Coordinate frames**: East-North-Up (ENU) for ground truth poses

## State Vector Structure

IEKF maintains state: `[Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i]`
- `Rot`: 3x3 rotation matrix (world to IMU)
- `v`: 3D velocity (world frame)
- `p`: 3D position (world frame)
- `b_omega`: gyro bias
- `b_acc`: accelerometer bias
- `Rot_c_i`: car to IMU rotation (extrinsic calibration)
- `t_c_i`: car to IMU translation (extrinsic calibration)

Covariance is 21-dimensional on the manifold tangent space.

## Training Configuration

Key hyperparameters in `train_torch_filter.py`:
- Learning rates: 1e-4 for all networks
- Loss: MSE on normalized relative pose errors
- Sequence length: 6000 timesteps (60s at 100 Hz)
- Default epochs: 400
- Gradient clipping: max norm of 1.0
- Results saved every epoch to `temp/iekfnets.p`
