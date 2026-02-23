# inertial-nav

A refactor of the [AI-IMU Dead-Reckoning](https://ieeexplore.ieee.org/document/9035481) paper by Brossard et al., with a modernized codebase layout and extensions for additional research experiments.

## What's here

- **Refactored IEKF + neural adapter** from the original paper, restructured for clarity and maintainability
- **Process & measurement model experiments** — exploring alternative noise parameterizations, covariance structures, and filter formulations
- **Latent variable world model** (see `world_models` branch) — extends the system toward learned latent state representations for richer scene modeling alongside the IEKF

## Original work

This builds on:
> M. Brossard, A. Barrau and S. Bonnabel, "AI-IMU Dead-Reckoning," *IEEE Transactions on Intelligent Vehicles*, 2020. [[paper](https://ieeexplore.ieee.org/document/9035481)] [[arXiv](https://arxiv.org/pdf/1904.06064.pdf)]

The original approach combines an Invariant Extended Kalman Filter (IEKF) with a CNN-based noise adapter to achieve 1.10% translational error on KITTI odometry using only IMU data.

## Quick start

```bash
cd src
python3 main_kitti.py
```

Download KITTI pickle data and pretrained weights (see https://github.com/user-attachments/files/17930695/data.zip) and place in `data/` directory.

## Requirements

```bash
pip install torch matplotlib numpy termcolor scipy navpy
```
