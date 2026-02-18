"""
Unit tests for PyTorch IEKF implementation.
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.torch_iekf import TorchIEKF, isclose


class TestTorchIEKFBasics:
    """Basic tests for PyTorch IEKF."""

    def test_instantiation(self):
        """Test that filter can be instantiated."""
        iekf = TorchIEKF()
        assert iekf is not None
        assert isinstance(iekf, torch.nn.Module)

    def test_parameters(self):
        """Test default parameters are set."""
        iekf = TorchIEKF()
        assert iekf.P_dim == 21
        assert iekf.Q_dim == 18
        assert isinstance(iekf.Q, torch.Tensor)

    def test_optional_networks(self):
        """Test that networks are optional."""
        iekf = TorchIEKF()
        assert iekf.initprocesscov_net is None
        assert iekf.mes_net is None

    def test_torch_identity_matrices(self):
        """Test PyTorch identity matrices."""
        iekf = TorchIEKF()
        assert isinstance(iekf.Id3, torch.Tensor)
        assert iekf.Id3.shape == (3, 3)
        assert torch.allclose(iekf.Id3, torch.eye(3).float())


class TestTorchIEKFGeometry:
    """Test PyTorch geometry methods."""

    def test_skew_torch(self):
        """Test skew-symmetric matrix."""
        v = torch.tensor([1.0, 2.0, 3.0]).float()
        S = TorchIEKF.skew_torch(v)

        # Check skew-symmetry
        assert torch.allclose(S.t(), -S)
        # Check trace is zero
        assert torch.isclose(torch.trace(S), torch.tensor(0.0).float())

    def test_so3exp_torch_identity(self):
        """Test SO(3) exp at identity."""
        phi = torch.zeros(3).float()
        Rot = TorchIEKF.so3exp_torch(phi)
        assert torch.allclose(Rot, torch.eye(3).float(), atol=1e-6)

    def test_so3exp_torch_properties(self):
        """Test SO(3) exp produces valid rotation."""
        phi = torch.tensor([0.1, 0.2, 0.3]).float()
        Rot = TorchIEKF.so3exp_torch(phi)

        # Check orthogonality
        assert torch.allclose(Rot @ Rot.t(), torch.eye(3).float(), atol=1e-6)
        # Check determinant
        assert torch.isclose(
            torch.det(Rot), torch.tensor(1.0).float(), atol=1e-6
        )

    def test_se23_exp_torch_identity(self):
        """Test SE_2(3) exp at identity."""
        xi = torch.zeros(9).float()
        Rot, x = TorchIEKF.se23_exp_torch(xi)

        assert torch.allclose(Rot, torch.eye(3).float())
        assert x.shape == (3, 2)

    def test_from_rpy_torch(self):
        """Test RPY to rotation matrix conversion."""
        roll = torch.tensor(0.1).float()
        pitch = torch.tensor(0.2).float()
        yaw = torch.tensor(0.3).float()

        Rot = TorchIEKF.from_rpy_torch(roll, pitch, yaw)

        # Check valid rotation matrix
        assert torch.allclose(Rot @ Rot.t(), torch.eye(3).float(), atol=1e-6)
        assert torch.isclose(
            torch.det(Rot), torch.tensor(1.0).float(), atol=1e-6
        )

    def test_normalize_rot_torch(self):
        """Test rotation normalization."""
        # Start with valid rotation
        Rot = TorchIEKF.from_rpy_torch(
            torch.tensor(0.1).float(),
            torch.tensor(0.2).float(),
            torch.tensor(0.3).float(),
        )

        # Add small noise
        Rot_noisy = Rot + 1e-5 * torch.randn(3, 3).float()

        # Normalize
        Rot_normalized = TorchIEKF.normalize_rot_torch(Rot_noisy)

        # Check properties restored
        assert torch.allclose(
            Rot_normalized @ Rot_normalized.t(),
            torch.eye(3).float(),
            atol=1e-5,
        )
        assert torch.isclose(
            torch.det(Rot_normalized), torch.tensor(1.0).float(), atol=1e-5
        )


class TestTorchIEKFInitialization:
    """Test filter initialization."""

    def test_init_covariance_without_network(self):
        """Test initial covariance without network."""
        iekf = TorchIEKF()
        P = iekf.init_covariance()

        assert P.shape == (21, 21)
        assert isinstance(P, torch.Tensor)
        # Check symmetry
        assert torch.allclose(P, P.t())

    def test_init_run(self):
        """Test state memory allocation via init_run."""
        iekf = TorchIEKF()
        N = 100
        dt = torch.ones(N - 1).float() * 0.01
        u = torch.zeros(N, 6).float()
        p_mes = torch.zeros(N, 3).float()
        v_mes = torch.zeros(N, 3).float()
        ang0 = torch.tensor([0.1, 0.2, 0.3]).float()

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = iekf.init_run(
            dt, u, p_mes, v_mes, N, ang0
        )

        assert Rot.shape == (N, 3, 3)
        assert v.shape == (N, 3)
        assert p.shape == (N, 3)
        # Check initial extrinsic calibration
        assert torch.allclose(Rot_c_i[0], torch.eye(3).float())


class TestTorchIEKFPropagation:
    """Test filter propagation."""

    def setup_method(self):
        """Set up test fixture."""
        self.iekf = TorchIEKF()

    def test_propagate_basic(self):
        """Test basic propagation step."""
        Rot = torch.eye(3).float()
        v = torch.zeros(3).float()
        p = torch.zeros(3).float()
        b_omega = torch.zeros(3).float()
        b_acc = torch.zeros(3).float()
        Rot_c_i = torch.eye(3).float()
        t_c_i = torch.zeros(3).float()
        P = self.iekf.init_covariance()

        u = torch.tensor([0, 0, 0, 0, 0, 9.80655]).float()
        dt = torch.tensor(0.01).float()

        (
            Rot_new,
            v_new,
            p_new,
            b_omega_new,
            b_acc_new,
            Rot_c_i_new,
            t_c_i_new,
            P_new,
        ) = self.iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt
        )

        # Check rotation matrix properties
        assert torch.allclose(
            Rot_new @ Rot_new.t(), torch.eye(3).float(), atol=1e-6
        )

        # Covariance should grow
        assert torch.trace(P_new) > torch.trace(P)

    def test_propagate_with_motion(self):
        """Test propagation with angular motion."""
        Rot = torch.eye(3).float()
        v = torch.zeros(3).float()
        p = torch.zeros(3).float()
        b_omega = torch.zeros(3).float()
        b_acc = torch.zeros(3).float()
        Rot_c_i = torch.eye(3).float()
        t_c_i = torch.zeros(3).float()
        P = self.iekf.init_covariance()

        # Angular velocity
        u = torch.tensor([0.1, 0, 0, 0, 0, 9.80655]).float()
        dt = torch.tensor(0.1).float()

        (
            Rot_new,
            v_new,
            p_new,
            b_omega_new,
            b_acc_new,
            Rot_c_i_new,
            t_c_i_new,
            P_new,
        ) = self.iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt
        )

        # Rotation should change
        assert not torch.allclose(Rot_new, Rot)
        # But still valid rotation
        assert torch.allclose(
            Rot_new @ Rot_new.t(), torch.eye(3).float(), atol=1e-6
        )

    def test_propagate_with_gyro_correction(self):
        """Test that gyro correction changes the rotation output."""
        Rot = torch.eye(3).float()
        v = torch.zeros(3).float()
        p = torch.zeros(3).float()
        b_omega = torch.zeros(3).float()
        b_acc = torch.zeros(3).float()
        Rot_c_i = torch.eye(3).float()
        t_c_i = torch.zeros(3).float()
        P = self.iekf.init_covariance()

        u = torch.tensor([0.1, 0.0, 0.0, 0, 0, 9.80655]).float()
        dt = torch.tensor(0.01).float()

        # Without gyro correction
        Rot_no_gc, *_ = self.iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt
        )

        # With gyro correction that partially cancels the angular velocity
        gc = torch.tensor([0.05, 0.0, 0.0]).float()
        Rot_gc, *_ = self.iekf.propagate(
            Rot,
            v,
            p,
            b_omega,
            b_acc,
            Rot_c_i,
            t_c_i,
            P,
            u,
            dt,
            gyro_correction=gc,
        )

        # Rotations should differ
        assert not torch.allclose(Rot_no_gc, Rot_gc, atol=1e-6)
        # Both should still be valid rotations
        assert torch.allclose(
            Rot_gc @ Rot_gc.t(), torch.eye(3).float(), atol=1e-6
        )

    def test_propagate_with_process_noise_scaling(self):
        """Test that process noise scaling changes the covariance output."""
        Rot = torch.eye(3).float()
        v = torch.zeros(3).float()
        p = torch.zeros(3).float()
        b_omega = torch.zeros(3).float()
        b_acc = torch.zeros(3).float()
        Rot_c_i = torch.eye(3).float()
        t_c_i = torch.zeros(3).float()
        P = self.iekf.init_covariance()

        u = torch.tensor([0.0, 0.0, 0.0, 0, 0, 9.80655]).float()
        dt = torch.tensor(0.01).float()

        # Without scaling
        *_, P_no_scale = self.iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt
        )

        # With 2x scaling on all Q blocks
        scaling = 2.0 * torch.ones(6).float()
        *_, P_scaled = self.iekf.propagate(
            Rot,
            v,
            p,
            b_omega,
            b_acc,
            Rot_c_i,
            t_c_i,
            P,
            u,
            dt,
            process_noise_scaling=scaling,
        )

        # Scaled covariance should be larger (more noise added)
        assert torch.trace(P_scaled) > torch.trace(P_no_scale)


class TestTorchIEKFUpdate:
    """Test filter update step."""

    def setup_method(self):
        """Set up test fixture."""
        self.iekf = TorchIEKF()

    def test_update_basic(self):
        """Test basic update step."""
        Rot = torch.eye(3).float()
        v = torch.tensor([1.0, 0.1, 0.05]).float()
        p = torch.tensor([10.0, 0.0, 0.0]).float()
        b_omega = torch.zeros(3).float()
        b_acc = torch.zeros(3).float()
        Rot_c_i = torch.eye(3).float()
        t_c_i = torch.zeros(3).float()
        P = self.iekf.init_covariance()

        u = torch.tensor([0, 0, 0, 0, 0, 9.80655]).float()
        measurement_cov = torch.tensor(
            [self.iekf.cov_lat, self.iekf.cov_up]
        ).float()

        (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        ) = self.iekf.update(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, 0, measurement_cov
        )

        # Lateral/vertical velocities should be reduced or stay same
        assert torch.abs(v_up[1]) <= torch.abs(v[1])
        assert torch.abs(v_up[2]) <= torch.abs(v[2])


class TestTorchIEKFIntegration:
    """Test full filter integration."""

    def test_run_short_sequence(self):
        """Test running filter on short sequence."""
        iekf = TorchIEKF()

        # Create synthetic data
        N = 30
        t = torch.linspace(0, 0.3, N).float()

        # Stationary vehicle
        u = torch.zeros(N, 6).float()
        u[:, 5] = 9.80655

        measurements_covs = torch.tensor(
            [[iekf.cov_lat, iekf.cov_up]] * N
        ).float()

        v_mes = torch.zeros(N, 3).float()
        p_mes = torch.zeros(N, 3).float()
        ang0 = torch.tensor([0.0, 0.0, 0.0]).float()

        # Run filter
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(
            t, u, measurements_covs, v_mes, p_mes, N, ang0
        )

        # Check shapes
        assert Rot.shape == (N, 3, 3)
        assert v.shape == (N, 3)
        assert p.shape == (N, 3)

        # Check all rotations are valid
        for i in range(N):
            assert torch.allclose(
                Rot[i] @ Rot[i].t(), torch.eye(3).float(), atol=1e-5
            )


class TestHelperFunctions:
    """Test helper functions."""

    def test_isclose_function(self):
        """Test isclose helper."""
        a = torch.tensor(1.0).float()
        b = torch.tensor(1.00001).float()  # Clearly different

        # Test that values within tolerance are close
        assert isclose(a, b, tol=1e-4)
        # Test that values outside tolerance are not close
        assert not isclose(a, b, tol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
