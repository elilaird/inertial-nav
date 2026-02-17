"""
Unit tests for NumPy IEKF implementation.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.numpy_iekf import NumPyIEKF
from src.core.base_filter import BaseFilter


class TestNumPyIEKFBasics:
    """Basic tests for NumPy IEKF."""

    def test_instantiation(self):
        """Test that filter can be instantiated."""
        iekf = NumPyIEKF()
        assert iekf is not None
        assert isinstance(iekf, BaseFilter)

    def test_parameters(self):
        """Test default parameters are set."""
        iekf = NumPyIEKF()
        assert iekf.P_dim == 21
        assert iekf.Q_dim == 18
        assert iekf.g is not None
        assert len(iekf.g) == 3

    def test_process_noise_matrix(self):
        """Test process noise covariance matrix."""
        iekf = NumPyIEKF()
        assert iekf.Q.shape == (18, 18)
        assert np.all(np.diag(iekf.Q) > 0)  # All diagonal elements positive
        # Check it's diagonal
        assert np.allclose(iekf.Q, np.diag(np.diag(iekf.Q)))

    def test_custom_parameters(self):
        """Test filter with custom parameters."""
        class CustomParams(NumPyIEKF.Parameters):
            cov_omega = 1e-2
            P_dim = 21

        iekf = NumPyIEKF(CustomParams)
        assert iekf.cov_omega == 1e-2


class TestNumPyIEKFInitialization:
    """Test filter initialization methods."""

    def test_init_covariance(self):
        """Test initial covariance matrix."""
        iekf = NumPyIEKF()
        P = iekf.init_covariance()

        assert P.shape == (21, 21)
        # Check symmetry
        assert np.allclose(P, P.T)
        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-10)

    def test_init_saved_state(self):
        """Test state memory allocation."""
        iekf = NumPyIEKF()
        N = 100
        dt = np.ones(N-1) * 0.01
        ang0 = np.array([0.1, 0.2, 0.3])

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.init_saved_state(dt, N, ang0)

        assert Rot.shape == (N, 3, 3)
        assert v.shape == (N, 3)
        assert p.shape == (N, 3)
        assert b_omega.shape == (N, 3)
        assert b_acc.shape == (N, 3)
        assert Rot_c_i.shape == (N, 3, 3)
        assert t_c_i.shape == (N, 3)
        # Check initial extrinsic calibration
        assert np.allclose(Rot_c_i[0], np.eye(3))


class TestNumPyIEKFPropagation:
    """Test filter propagation step."""

    def setup_method(self):
        """Set up test fixture."""
        self.iekf = NumPyIEKF()

    def test_propagate_identity(self):
        """Test propagation with zero motion."""
        # Initial state
        Rot = np.eye(3)
        v = np.zeros(3)
        p = np.zeros(3)
        b_omega = np.zeros(3)
        b_acc = np.zeros(3)
        Rot_c_i = np.eye(3)
        t_c_i = np.zeros(3)
        P = self.iekf.init_covariance()

        # Zero IMU input (hovering in gravity)
        u = np.array([0, 0, 0, 0, 0, 9.80655])  # Only gravity compensation
        dt = 0.01

        Rot_new, v_new, p_new, b_omega_new, b_acc_new, Rot_c_i_new, t_c_i_new, P_new = \
            self.iekf.propagate(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt)

        # With gravity-compensating acceleration, velocity should remain near zero
        assert np.linalg.norm(v_new) < 0.01
        # Position should barely change
        assert np.linalg.norm(p_new) < 0.001
        # Covariance should grow
        assert np.trace(P_new) > np.trace(P)

    def test_propagate_rotation(self):
        """Test propagation with rotation."""
        Rot = np.eye(3)
        v = np.zeros(3)
        p = np.zeros(3)
        b_omega = np.zeros(3)
        b_acc = np.zeros(3)
        Rot_c_i = np.eye(3)
        t_c_i = np.zeros(3)
        P = self.iekf.init_covariance()

        # Angular velocity around z-axis
        omega_z = 0.1  # rad/s
        u = np.array([0, 0, omega_z, 0, 0, 9.80655])
        dt = 0.1

        Rot_new, v_new, p_new, b_omega_new, b_acc_new, Rot_c_i_new, t_c_i_new, P_new = \
            self.iekf.propagate(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt)

        # Rotation should have changed
        assert not np.allclose(Rot_new, Rot)
        # Should still be a valid rotation matrix
        assert np.allclose(Rot_new @ Rot_new.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(Rot_new), 1.0, atol=1e-6)


class TestNumPyIEKFUpdate:
    """Test filter update step."""

    def setup_method(self):
        """Set up test fixture."""
        self.iekf = NumPyIEKF()

    def test_update_basic(self):
        """Test basic update step."""
        # State with some velocity
        Rot = np.eye(3)
        v = np.array([1.0, 0.1, 0.05])  # Small lateral/vertical velocity
        p = np.array([10.0, 0.0, 0.0])
        b_omega = np.zeros(3)
        b_acc = np.zeros(3)
        Rot_c_i = np.eye(3)
        t_c_i = np.zeros(3)
        P = self.iekf.init_covariance()

        u = np.array([0, 0, 0, 0, 0, 9.80655])
        measurement_cov = np.array([self.iekf.cov_lat, self.iekf.cov_up])

        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.iekf.update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, 0, measurement_cov)

        # Lateral and vertical velocity should be reduced or stay same (zero velocity constraint)
        assert np.abs(v_up[1]) <= np.abs(v[1])
        assert np.abs(v_up[2]) <= np.abs(v[2])

        # Covariance should be reduced (information gained)
        assert np.trace(P_up) < np.trace(P)


class TestNumPyIEKFGeometry:
    """Test geometry helper methods."""

    def test_se23_exp_identity(self):
        """Test SE_2(3) exponential at identity."""
        xi = np.zeros(9)
        Rot, x = NumPyIEKF.se23_exp(xi)

        assert np.allclose(Rot, np.eye(3))
        assert x.shape == (3, 2)

    def test_se23_exp_rotation_only(self):
        """Test SE_2(3) exp with rotation only."""
        xi = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0])
        Rot, x = NumPyIEKF.se23_exp(xi)

        # Check rotation matrix properties
        assert np.allclose(Rot @ Rot.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(Rot), 1.0, atol=1e-10)

    def test_rot_from_2_vectors(self):
        """Test rotation from two vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])

        Rot = NumPyIEKF.rot_from_2_vectors(v1, v2)

        # Check that Rot @ v1 is parallel to v2
        result = Rot @ v1
        result_normalized = result / np.linalg.norm(result)
        assert np.allclose(result_normalized, v2, atol=1e-6)

        # Check valid rotation matrix
        assert np.allclose(Rot @ Rot.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(Rot), 1.0, atol=1e-10)


class TestNumPyIEKFRun:
    """Test full filter run."""

    def test_run_short_sequence(self):
        """Test running filter on short sequence."""
        iekf = NumPyIEKF()

        # Create synthetic data
        N = 50
        t = np.linspace(0, 0.5, N)  # 0.5 seconds at 100 Hz

        # Stationary vehicle with gravity compensation
        u = np.zeros((N, 6))
        u[:, 5] = 9.80655  # Gravity in accelerometer

        # Constant measurement covariances
        measurements_covs = np.tile([iekf.cov_lat, iekf.cov_up], (N, 1))

        # Ground truth (stationary)
        v_mes = np.zeros((N, 3))
        p_mes = np.zeros((N, 3))
        ang0 = np.array([0.0, 0.0, 0.0])

        # Run filter
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = \
            iekf.run(t, u, measurements_covs, v_mes, p_mes, N, ang0)

        # Check outputs have correct shape
        assert Rot.shape == (N, 3, 3)
        assert v.shape == (N, 3)
        assert p.shape == (N, 3)
        assert b_omega.shape == (N, 3)
        assert b_acc.shape == (N, 3)

        # For stationary vehicle, position drift should be small
        assert np.linalg.norm(p[-1]) < 1.0  # Less than 1 meter drift


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
