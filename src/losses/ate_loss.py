"""
Absolute Trajectory Error (ATE) loss.

Computes the RMSE between predicted and ground truth positions,
optionally after Umeyama alignment (SE(3) registration).
"""

import torch
import torch.nn as nn

from src.losses.base_loss import TrajectoryLoss


class ATELoss(TrajectoryLoss):
    """
    Absolute Trajectory Error loss.

    Computes MSE between predicted and ground truth positions.
    Optionally applies Umeyama alignment before computing the error,
    which is standard for ATE evaluation but not for training
    (alignment is non-differentiable).

    Args:
        cfg: Config dict with keys:
            - align: Whether to apply Umeyama alignment (default: False)
            - reduction: "mean" or "sum" (default: "mean")
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.align = cfg.get("align", False)
        reduction = cfg.get("reduction", "mean")
        self.criterion = nn.MSELoss(reduction=reduction)

    def precompute(self, Rot_gt, p_gt):
        """No precomputation needed for ATE."""
        return None

    def forward(self, Rot_pred, p_pred, Rot_gt, p_gt, **kwargs):
        """
        Compute ATE loss.

        Args:
            Rot_pred: Predicted rotations (N, 3, 3) - unused
            p_pred: Predicted positions (N, 3)
            Rot_gt: Ground truth rotations (N, 3, 3) - unused
            p_gt: Ground truth positions (N, 3)

        Returns:
            Scalar loss tensor (MSE of position errors).
        """
        if self.align:
            p_pred_aligned = self._umeyama_align_torch(p_pred, p_gt)
            return self.criterion(p_pred_aligned, p_gt)
        return self.criterion(p_pred, p_gt)

    @staticmethod
    def _umeyama_align_torch(src, dst):
        """
        Simple Umeyama alignment (translation + rotation, no scale).

        Uses SVD to find the optimal rigid transform from src to dst.
        Differentiable through PyTorch's SVD.

        Args:
            src: Source points (N, 3)
            dst: Destination points (N, 3)

        Returns:
            Aligned source points (N, 3)
        """
        src_mean = src.mean(dim=0)
        dst_mean = dst.mean(dim=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.t().mm(dst_centered)
        U, _, Vt = torch.linalg.svd(H)

        d = torch.det(Vt.t().mm(U.t()))
        S = torch.eye(3, dtype=src.dtype, device=src.device)
        S[2, 2] = d.sign()

        R = Vt.t().mm(S).mm(U.t())
        t = dst_mean - R.mv(src_mean)

        return src.mm(R.t()) + t
