"""Tests for loss functions (Phase 3)."""

import pytest
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.losses import get_loss, list_losses
from src.losses.base_loss import TrajectoryLoss
from src.losses.rpe_loss import RPELoss
from src.losses.ate_loss import ATELoss


# ==================== Loss Registry Tests ====================

class TestLossRegistry:
    def test_list_losses(self):
        names = list_losses()
        assert "RPELoss" in names
        assert "ATELoss" in names

    def test_get_loss_rpe(self):
        loss = get_loss("RPELoss")
        assert isinstance(loss, RPELoss)

    def test_get_loss_ate(self):
        loss = get_loss("ATELoss")
        assert isinstance(loss, ATELoss)

    def test_get_loss_unknown(self):
        with pytest.raises(KeyError, match="not found"):
            get_loss("NonExistentLoss")

    def test_get_loss_with_cfg(self):
        loss = get_loss("RPELoss", cfg={"criterion": "l1"})
        assert isinstance(loss, RPELoss)


# ==================== Helper: Create Fake Trajectory ====================

def make_straight_trajectory(N=5000, dt=0.01):
    """Create a straight-line trajectory for testing."""
    # Moving at 10 m/s along x-axis
    speed = 10.0
    t = torch.arange(N).float() * dt
    p = torch.zeros(N, 3).float()
    p[:, 0] = speed * t  # x = speed * t
    v = torch.zeros(N, 3).float()
    v[:, 0] = speed

    # Identity rotations
    Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1).clone()

    return Rot, p, v, t


def make_circular_trajectory(N=10000, dt=0.01, radius=100.0):
    """Create a circular trajectory for testing."""
    t = torch.arange(N).float() * dt
    omega = 2 * np.pi / (N * dt)  # one full circle

    p = torch.zeros(N, 3).float()
    p[:, 0] = radius * torch.cos(omega * t)
    p[:, 1] = radius * torch.sin(omega * t)

    Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1).clone()
    for i in range(N):
        angle = omega * t[i]
        c, s = torch.cos(angle), torch.sin(angle)
        Rot[i, 0, 0] = c
        Rot[i, 0, 1] = -s
        Rot[i, 1, 0] = s
        Rot[i, 1, 1] = c

    return Rot, p, Rot, t


# ==================== RPE Loss Tests ====================

class TestRPELoss:
    def test_precompute_shapes(self):
        Rot, p, v, t = make_straight_trajectory(N=5000)
        loss = RPELoss()
        list_rpe = loss.precompute(Rot, p)
        assert len(list_rpe) == 3
        assert len(list_rpe[0]) > 0  # has index pairs
        assert len(list_rpe[0]) == len(list_rpe[1])
        assert list_rpe[2].shape[1] == 3

    def test_precompute_short_trajectory(self):
        """Very short trajectory should still work (may have no pairs)."""
        Rot = torch.eye(3).float().unsqueeze(0).expand(50, -1, -1).clone()
        p = torch.zeros(50, 3).float()
        loss = RPELoss()
        list_rpe = loss.precompute(Rot, p)
        assert len(list_rpe) == 3

    def test_forward_perfect_prediction(self):
        """Zero loss when prediction matches ground truth."""
        Rot, p, v, t = make_straight_trajectory(N=5000)
        loss_fn = RPELoss()
        list_rpe = loss_fn.precompute(Rot, p)

        loss = loss_fn(Rot, p, Rot, p, list_rpe=list_rpe, N0=0)
        if loss != -1:
            assert loss.item() < 1e-6

    def test_forward_with_error(self):
        """Non-zero loss when prediction differs from ground truth."""
        Rot, p, v, t = make_straight_trajectory(N=5000)
        loss_fn = RPELoss()
        list_rpe = loss_fn.precompute(Rot, p)

        # Add position error
        p_noisy = p.clone()
        p_noisy[:, 0] += torch.randn(p.shape[0]).float() * 5.0

        loss = loss_fn(Rot, p_noisy, Rot, p, list_rpe=list_rpe, N0=0)
        if loss != -1:
            assert loss.item() > 0

    def test_forward_subsequence(self):
        """Test with N0 offset for subsequence training."""
        Rot, p, v, t = make_straight_trajectory(N=5000)
        loss_fn = RPELoss()
        list_rpe = loss_fn.precompute(Rot, p)

        # Use a subsequence starting at N0=1000
        N0 = 1000
        N_end = 4000
        loss = loss_fn(Rot[N0:N_end], p[N0:N_end] - p[N0], Rot, p,
                       list_rpe=list_rpe, N0=N0)
        # Should compute without error (may return -1 if no valid pairs)
        assert loss == -1 or isinstance(loss, torch.Tensor)

    def test_gradient_flow(self):
        """Verify gradients flow through RPE loss."""
        Rot, p, v, t = make_straight_trajectory(N=5000)
        loss_fn = RPELoss()
        list_rpe = loss_fn.precompute(Rot, p)

        p_pred = p.clone().requires_grad_(True)
        loss = loss_fn(Rot, p_pred, Rot, p, list_rpe=list_rpe, N0=0)
        if loss != -1:
            loss.backward()
            assert p_pred.grad is not None

    def test_l1_criterion(self):
        loss_fn = RPELoss(cfg={"criterion": "l1"})
        Rot, p, v, t = make_straight_trajectory(N=5000)
        list_rpe = loss_fn.precompute(Rot, p)
        loss = loss_fn(Rot, p, Rot, p, list_rpe=list_rpe)
        if loss != -1:
            assert loss.item() < 1e-6

    def test_custom_distance_windows(self):
        loss_fn = RPELoss(cfg={"distance_windows": [50, 100]})
        assert loss_fn.distance_windows == [50, 100]


# ==================== ATE Loss Tests ====================

class TestATELoss:
    def test_forward_perfect(self):
        N = 100
        Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1)
        p = torch.randn(N, 3).float()
        loss_fn = ATELoss()
        loss = loss_fn(Rot, p, Rot, p)
        assert loss.item() < 1e-10

    def test_forward_with_error(self):
        N = 100
        Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1)
        p_gt = torch.randn(N, 3).float()
        p_pred = p_gt + torch.randn(N, 3).float() * 0.1
        loss_fn = ATELoss()
        loss = loss_fn(Rot, p_pred, Rot, p_gt)
        assert loss.item() > 0

    def test_forward_with_alignment(self):
        N = 100
        Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1)
        p_gt = torch.randn(N, 3).float()
        # Translate prediction — alignment should reduce error
        p_pred = p_gt + torch.tensor([10.0, 0.0, 0.0]).float()
        loss_fn_no_align = ATELoss(cfg={"align": False})
        loss_fn_align = ATELoss(cfg={"align": True})
        loss_no = loss_fn_no_align(Rot, p_pred, Rot, p_gt)
        loss_yes = loss_fn_align(Rot, p_pred, Rot, p_gt)
        assert loss_yes.item() < loss_no.item()

    def test_gradient_flow(self):
        N = 50
        Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1)
        p_gt = torch.randn(N, 3).float()
        p_pred = p_gt.clone().requires_grad_(True)
        loss_fn = ATELoss()
        loss = loss_fn(Rot, p_pred, Rot, p_gt)
        loss.backward()
        assert p_pred.grad is not None

    def test_precompute_returns_none(self):
        loss_fn = ATELoss()
        result = loss_fn.precompute(None, None)
        assert result is None

    def test_reduction_sum(self):
        N = 50
        Rot = torch.eye(3).float().unsqueeze(0).expand(N, -1, -1)
        p_gt = torch.randn(N, 3).float()
        p_pred = p_gt + 0.1

        loss_mean = ATELoss(cfg={"reduction": "mean"})(Rot, p_pred, Rot, p_gt)
        loss_sum = ATELoss(cfg={"reduction": "sum"})(Rot, p_pred, Rot, p_gt)
        assert loss_sum.item() > loss_mean.item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
