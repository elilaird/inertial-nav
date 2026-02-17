"""Tests for evaluation metrics and visualization (Phase 5)."""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metrics import compute_rpe, compute_ate, compute_orientation_error
from src.evaluation.visualization import (
    plot_trajectory_2d, plot_trajectory_3d,
    plot_error_timeline, plot_covariance_timeline, plot_state_estimates,
)


# ==================== Metrics Tests ====================

class TestComputeRPE:
    def _make_straight_line(self, N=10000, speed=10.0, dt=0.01):
        """Create a straight-line trajectory at constant speed."""
        p = np.zeros((N, 3))
        p[:, 0] = np.arange(N) * speed * dt
        Rot = np.tile(np.eye(3), (N, 1, 1))
        return Rot, p

    def test_perfect_prediction(self):
        Rot_gt, p_gt = self._make_straight_line()
        rpe = compute_rpe(Rot_gt, p_gt, Rot_gt, p_gt)
        assert rpe['mean'] == pytest.approx(0.0, abs=1e-6)

    def test_noisy_prediction(self):
        Rot_gt, p_gt = self._make_straight_line()
        p_pred = p_gt + np.random.randn(*p_gt.shape) * 0.1
        rpe = compute_rpe(Rot_gt, p_pred, Rot_gt, p_gt)
        assert rpe['mean'] > 0
        assert 'rmse' in rpe
        assert 'std' in rpe

    def test_per_distance_keys(self):
        Rot_gt, p_gt = self._make_straight_line()
        rpe = compute_rpe(Rot_gt, p_gt, Rot_gt, p_gt)
        for d in [100, 200, 300, 400, 500, 600, 700, 800]:
            assert f'rpe_{d}m' in rpe

    def test_short_sequence_returns_nan(self):
        N = 100
        Rot = np.tile(np.eye(3), (N, 1, 1))
        p = np.zeros((N, 3))
        rpe = compute_rpe(Rot, p, Rot, p)
        assert np.isnan(rpe['mean'])

    def test_custom_distance_windows(self):
        Rot_gt, p_gt = self._make_straight_line()
        rpe = compute_rpe(Rot_gt, p_gt, Rot_gt, p_gt, distance_windows=[100, 200])
        assert 'rpe_100m' in rpe
        assert 'rpe_200m' in rpe
        assert 'rpe_300m' not in rpe


class TestComputeATE:
    def test_perfect_prediction(self):
        N = 1000
        p_gt = np.random.randn(N, 3).cumsum(axis=0)
        ate = compute_ate(p_gt, p_gt, align=False)
        assert ate['mean'] == pytest.approx(0.0, abs=1e-6)

    def test_with_alignment(self):
        N = 1000
        p_gt = np.random.randn(N, 3).cumsum(axis=0)
        # Add offset and small rotation
        p_pred = p_gt + np.array([10.0, 5.0, 0.0])
        ate = compute_ate(p_pred, p_gt, align=True)
        # After alignment, error should be small
        assert ate['mean'] < 1.0

    def test_without_alignment(self):
        N = 1000
        p_gt = np.random.randn(N, 3).cumsum(axis=0)
        p_pred = p_gt + np.array([10.0, 5.0, 0.0])
        ate = compute_ate(p_pred, p_gt, align=False)
        # Without alignment, error should be ~offset magnitude
        assert ate['mean'] > 5.0

    def test_output_keys(self):
        N = 100
        p = np.random.randn(N, 3)
        ate = compute_ate(p, p, align=False)
        assert all(k in ate for k in ['mean', 'std', 'rmse', 'median', 'max'])


class TestComputeOrientationError:
    def test_perfect_prediction(self):
        N = 100
        Rot = np.tile(np.eye(3), (N, 1, 1))
        err = compute_orientation_error(Rot, Rot)
        assert err['mean_deg'] == pytest.approx(0.0, abs=1e-6)

    def test_known_rotation(self):
        N = 50
        Rot_gt = np.tile(np.eye(3), (N, 1, 1))
        # 90 degree rotation around z-axis
        Rot_pred = np.tile(np.eye(3), (N, 1, 1))
        Rot_pred[:, 0, 0] = 0
        Rot_pred[:, 0, 1] = -1
        Rot_pred[:, 1, 0] = 1
        Rot_pred[:, 1, 1] = 0
        err = compute_orientation_error(Rot_pred, Rot_gt)
        assert err['mean_deg'] == pytest.approx(90.0, abs=1.0)

    def test_output_keys(self):
        N = 10
        Rot = np.tile(np.eye(3), (N, 1, 1))
        err = compute_orientation_error(Rot, Rot)
        assert all(k in err for k in ['mean_deg', 'std_deg', 'rmse_deg', 'max_deg'])


# ==================== Visualization Tests ====================

class TestVisualization:
    def _make_trajectory(self, N=100):
        t = np.linspace(0, 10, N)
        p_gt = np.column_stack([np.cos(t), np.sin(t), np.zeros(N)]) * 100
        p_pred = p_gt + np.random.randn(N, 3) * 2
        return p_pred, p_gt, t

    def test_plot_trajectory_2d(self):
        p_pred, p_gt, _ = self._make_trajectory()
        fig = plot_trajectory_2d(p_pred, p_gt, seq_name="test_seq")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_trajectory_3d(self):
        p_pred, p_gt, _ = self._make_trajectory()
        fig = plot_trajectory_3d(p_pred, p_gt, seq_name="test_seq")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_error_timeline_1d(self):
        _, _, t = self._make_trajectory()
        errors = np.random.rand(len(t))
        fig = plot_error_timeline(errors, timestamps=t, seq_name="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_error_timeline_3d(self):
        _, _, t = self._make_trajectory()
        errors = np.random.rand(len(t), 3)
        fig = plot_error_timeline(errors, timestamps=t, seq_name="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_error_timeline_no_timestamps(self):
        errors = np.random.rand(50)
        fig = plot_error_timeline(errors)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_covariance_timeline(self):
        N = 100
        covs = np.abs(np.random.randn(N, 2)) * 0.1
        t = np.linspace(0, 10, N)
        fig = plot_covariance_timeline(covs, timestamps=t, seq_name="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_state_estimates_position_only(self):
        p_pred, p_gt, t = self._make_trajectory()
        fig = plot_state_estimates(p_pred, p_gt, timestamps=t, seq_name="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_state_estimates_with_velocity(self):
        p_pred, p_gt, t = self._make_trajectory()
        v_gt = np.gradient(p_gt, axis=0) * 10
        v_pred = v_gt + np.random.randn(*v_gt.shape) * 0.5
        fig = plot_state_estimates(p_pred, p_gt, v_pred, v_gt, t, "test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
