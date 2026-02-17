"""
Abstract base class for trajectory loss functions.

All loss functions operate on predicted and ground truth
rotation matrices and positions from the IEKF filter output.
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class TrajectoryLoss(nn.Module, ABC):
    """
    Abstract trajectory loss function.

    Args:
        cfg: Loss-specific configuration.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}

    @abstractmethod
    def forward(self, Rot_pred, p_pred, Rot_gt, p_gt, **kwargs):
        """
        Compute loss between predicted and ground truth trajectories.

        Args:
            Rot_pred: Predicted rotations (N, 3, 3)
            p_pred: Predicted positions (N, 3)
            Rot_gt: Ground truth rotations (N, 3, 3)
            p_gt: Ground truth positions (N, 3)

        Returns:
            Scalar loss tensor.
        """
        pass

    @abstractmethod
    def precompute(self, Rot_gt, p_gt):
        """
        Precompute ground truth data needed for loss computation.

        Called once per sequence before training to cache
        expensive computations (e.g., relative pose pairs for RPE).

        Args:
            Rot_gt: Ground truth rotations (N, 3, 3)
            p_gt: Ground truth positions (N, 3)

        Returns:
            Precomputed data structure for this sequence.
        """
        pass
