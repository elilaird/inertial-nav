"""
Relative Pose Error (RPE) loss for trajectory evaluation and training.

The RPE loss computes errors between relative pose displacements at
various distance windows (100m, 200m, ..., 800m). This is the standard
evaluation metric for visual odometry benchmarks (KITTI).

During training, the loss is computed as MSE between normalized
relative displacements in the local frame.
"""

import torch
import torch.nn as nn
import numpy as np

from src.losses.base_loss import TrajectoryLoss


class RPELoss(TrajectoryLoss):
    """
    Relative Pose Error loss.

    Computes MSE between predicted and ground truth relative position
    displacements, normalized by distance. Matches the KITTI odometry
    benchmark evaluation protocol.

    Args:
        cfg: Config dict with keys:
            - criterion: "mse" or "l1" (default: "mse")
            - downsample_rate: Downsample factor from raw rate (default: 10, i.e. 100Hz -> 10Hz)
            - step_size: Step size for sampling start indices (default: 10)
            - distance_windows: List of distances in meters (default: [100..800])
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}

        criterion_name = cfg.get("criterion", "mse")
        if criterion_name == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif criterion_name == "l1":
            self.criterion = nn.L1Loss(reduction="sum")
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

        self.downsample_rate = cfg.get("downsample_rate", 10)
        self.step_size = cfg.get("step_size", 10)
        self.distance_windows = cfg.get("distance_windows",
            [100, 200, 300, 400, 500, 600, 700, 800])

    def precompute(self, Rot_gt, p_gt):
        """
        Precompute ground truth relative pose pairs for RPE evaluation.

        Downsamples to 10Hz, then finds pairs of indices at each
        distance window along the trajectory.

        Args:
            Rot_gt: Ground truth rotations (N, 3, 3)
            p_gt: Ground truth positions (N, 3)

        Returns:
            list_rpe: [idx_starts, idx_ends, delta_p_gt]
                - idx_starts: Start indices (at downsampled rate)
                - idx_ends: End indices (at downsampled rate)
                - delta_p_gt: Ground truth relative displacements in local frame
        """
        rate = self.downsample_rate
        Rot = Rot_gt[::rate]
        p = p_gt[::rate]

        list_rpe = [[], [], []]  # [idx_0, idx_end, delta_p]

        # Compute cumulative distances
        distances = np.zeros(p.shape[0])
        dp = p[1:] - p[:-1]
        distances[1:] = dp.norm(dim=1).cumsum(0).numpy()

        k_max = int(Rot.shape[0] / self.step_size) - 1

        for k in range(0, k_max):
            idx_0 = k * self.step_size
            for seq_length in self.distance_windows:
                if seq_length + distances[idx_0] > distances[-1]:
                    continue
                idx_shift = np.searchsorted(distances[idx_0:],
                                            distances[idx_0] + seq_length)
                idx_end = idx_0 + idx_shift

                list_rpe[0].append(idx_0)
                list_rpe[1].append(idx_end)

        if len(list_rpe[0]) == 0:
            list_rpe[2] = torch.zeros(0, 3)
            return list_rpe

        idxs_0 = list_rpe[0]
        idxs_end = list_rpe[1]

        # Compute ground truth relative displacements in local frame
        diff = (p[idxs_end] - p[idxs_0]).to(Rot.dtype)
        delta_p = Rot[idxs_0].transpose(-1, -2).matmul(
            diff.unsqueeze(-1)).squeeze()
        list_rpe[2] = delta_p

        return list_rpe

    def forward(self, Rot_pred, p_pred, Rot_gt, p_gt, list_rpe=None, N0=0):
        """
        Compute RPE loss between predicted and ground truth trajectories.

        Args:
            Rot_pred: Predicted rotations (N, 3, 3)
            p_pred: Predicted positions (N, 3)
            Rot_gt: Ground truth rotations (N, 3, 3) - unused, precomputed
            p_gt: Ground truth positions (N, 3) - unused, precomputed
            list_rpe: Precomputed RPE data from precompute().
            N0: Start index offset for subsequence training.

        Returns:
            Scalar loss tensor, or -1 if no valid pairs.
        """
        if list_rpe is None:
            list_rpe = self.precompute(Rot_gt, p_gt)

        delta_p, delta_p_gt = self._compute_predicted_rpe(
            Rot_pred, p_pred, list_rpe, N0
        )

        if delta_p is None:
            return -1

        return self.criterion(delta_p, delta_p_gt)

    def _compute_predicted_rpe(self, Rot, p, list_rpe, N0=0):
        """
        Compute predicted relative displacements matching precomputed GT pairs.

        Args:
            Rot: Predicted rotations (N, 3, 3)
            p: Predicted positions (N, 3)
            list_rpe: Precomputed [idx_starts, idx_ends, delta_p_gt]
            N0: Offset for subsequence indexing.

        Returns:
            Tuple of (delta_p_pred, delta_p_gt) normalized by distance,
            or (None, None) if no valid pairs.
        """
        rate = self.downsample_rate
        N = p.shape[0]
        Rot_ds = Rot[::rate]
        p_ds = p[::rate]

        idxs_0 = torch.Tensor(list_rpe[0]).clone().long() - int(N0 / rate)
        idxs_end = torch.Tensor(list_rpe[1]).clone().long() - int(N0 / rate)
        delta_p_gt = list_rpe[2]

        # Filter to valid indices within current subsequence
        valid = torch.ones(idxs_0.shape[0], dtype=torch.bool)
        valid[idxs_0 < 0] = False
        valid[idxs_end >= int(N / rate)] = False

        delta_p_gt = delta_p_gt[valid]
        idxs_end = idxs_end[valid]
        idxs_0 = idxs_0[valid]

        if len(idxs_0) == 0:
            return None, None

        # Compute predicted relative displacements in local frame
        delta_p = Rot_ds[idxs_0].transpose(-1, -2).matmul(
            (p_ds[idxs_end] - p_ds[idxs_0]).unsqueeze(-1)).squeeze()

        # Normalize by distance
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        return delta_p.double() / distance.double(), delta_p_gt.double() / distance.double()
