"""
Composable data transforms for IMU sequences.

These transforms follow the PyTorch-style composition pattern and can be
chained together for data augmentation and preprocessing.
"""

import torch
import numpy as np


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class AddIMUNoise:
    """
    Add random noise to IMU measurements for data augmentation.

    Adds both per-sample noise and a constant bias perturbation.

    Args:
        sigma_gyro: Gyroscope noise standard deviation.
        sigma_acc: Accelerometer noise standard deviation.
        sigma_b_gyro: Gyroscope bias noise standard deviation.
        sigma_b_acc: Accelerometer bias noise standard deviation.
    """

    def __init__(self, sigma_gyro=1e-4, sigma_acc=1e-4,
                 sigma_b_gyro=1e-5, sigma_b_acc=1e-4):
        self.sigma_gyro = sigma_gyro
        self.sigma_acc = sigma_acc
        self.sigma_b_gyro = sigma_b_gyro
        self.sigma_b_acc = sigma_b_acc

    def __call__(self, data):
        u = data['u'].clone()
        w = torch.randn_like(u[:, :6])
        w_b = torch.randn_like(u[0, :6])
        w[:, :3] *= self.sigma_gyro
        w[:, 3:6] *= self.sigma_acc
        w_b[:3] *= self.sigma_b_gyro
        w_b[3:6] *= self.sigma_b_acc
        u[:, :6] += w + w_b
        data['u'] = u
        return data


class NormalizeIMU:
    """
    Normalize IMU inputs using precomputed mean and standard deviation.

    Args:
        u_loc: Mean of IMU inputs (6,).
        u_std: Standard deviation of IMU inputs (6,).
    """

    def __init__(self, u_loc, u_std):
        self.u_loc = u_loc
        self.u_std = u_std

    def __call__(self, data):
        u = data['u']
        data['u_normalized'] = (u - self.u_loc) / self.u_std
        return data


class RandomSubsequenceSampler:
    """
    Extract a random contiguous subsequence of fixed length.

    Samples a random start index (aligned to 10-sample boundaries)
    and extracts a subsequence of length seq_dim.

    Args:
        seq_dim: Length of subsequence to extract.
    """

    def __init__(self, seq_dim):
        self.seq_dim = seq_dim

    def __call__(self, data):
        N = data['u'].shape[0]
        if self.seq_dim is None or N <= self.seq_dim:
            data['N0'] = 0
            return data

        N0 = 10 * int(np.random.randint(0, (N - self.seq_dim) / 10))
        N_end = N0 + self.seq_dim

        for key in ['t', 'u', 'ang_gt', 'v_gt']:
            if key in data:
                data[key] = data[key][N0:N_end]

        if 'p_gt' in data:
            data['p_gt'] = data['p_gt'][N0:N_end] - data['p_gt'][N0]

        data['N0'] = N0
        return data


class ToTensor:
    """Convert numpy arrays in data dict to PyTorch tensors."""

    def __init__(self, dtype=torch.float64):
        self.dtype = dtype

    def __call__(self, data):
        for key in ['t', 'u', 'ang_gt', 'p_gt', 'v_gt']:
            if key in data and isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key]).to(self.dtype)
            elif key in data and isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.dtype)
        return data
