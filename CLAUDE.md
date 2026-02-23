# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration
### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution
### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project
### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it
### 6. Autonomous Bug Fizing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how
## Task Management
1. **Plan First**: Write plan to tasks/todo md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**; Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md*
6. **Capture Lessons**: Update 'tasks/lessons md after corrections
## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Project Overview

AI-IMU Dead-Reckoning system implementing accurate vehicle localization using only Inertial Measurement Unit (IMU) data. The approach combines an Invariant Extended Kalman Filter (IEKF) with deep neural networks that dynamically adapt noise parameters in real-time. Achieves 1.10% translational error on KITTI odometry sequences.

## Core Architecture

The system consists of two main blocks:

1. **IEKF Filter Block** (`src/core/torch_iekf.py`)
   - Integrates inertial measurements (gyroscope and accelerometer)
   - Exploits zero lateral and vertical velocity constraints
   - Estimates 3D position, velocity, orientation, and IMU biases
   - Pure PyTorch implementation with CUDA support

2. **Neural Network Adapter Block** (`src/models/`)
   - `InitProcessCovNet`: Learns initial state and process noise covariances
   - `MeasurementCovNet`: CNN that converts raw IMU signals into measurement covariance matrices
   - PyTorch-based for training and inference
   - No knowledge of state estimates required - operates directly on IMU signals

The neural networks predict covariances which are used by the IEKF filter during both training and testing.

## Key Files

- `main_kitti.py` - Main entry point defining the KITTI dataset class, parameters, and workflow control
- `dataset.py` - Base dataset class handling data loading, normalization, and noise injection
- `utils_torch_filter.py` - PyTorch IEKF with neural network components (training & inference, legacy)
- `train_torch_filter.py` - Training loop using relative pose error (RPE) loss (legacy)
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
4. **Testing**: Neural nets predict covariances → TorchIEKF runs filter (with `torch.no_grad()`) → Results saved to `results/`
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
