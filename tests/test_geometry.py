"""
Unit tests for geometry utilities.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.geometry import (
    so3_exp, so3_left_jacobian, normalize_rot,
    from_rpy, to_rpy, rotx, roty, rotz,
    skew_symmetric, umeyama_alignment
)


class TestSO3Operations:
    """Tests for SO(3) operations"""

    def test_so3_exp_identity(self):
        """Test that exp(0) = Identity"""
        phi = np.zeros(3)
        R = so3_exp(phi)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_so3_exp_small_angle(self):
        """Test SO(3) exp for small angles (Taylor expansion branch)"""
        phi = np.array([1e-9, 2e-9, 3e-9])
        R = so3_exp(phi)
        # Should be close to identity
        assert np.linalg.det(R) > 0.99
        assert np.allclose(R @ R.T, np.eye(3))

    def test_so3_exp_properties(self):
        """Test that exp produces valid rotation matrices"""
        phi = np.array([0.1, 0.2, 0.3])
        R = so3_exp(phi)

        # Check orthogonality: R @ R.T = I
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)

        # Check determinant = 1
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)

    def test_so3_left_jacobian_identity(self):
        """Test left Jacobian at identity"""
        phi = np.zeros(3)
        J = so3_left_jacobian(phi)
        np.testing.assert_array_almost_equal(J, np.eye(3))

    def test_so3_left_jacobian_small_angle(self):
        """Test left Jacobian for small angles"""
        phi = np.array([1e-9, 1e-9, 1e-9])
        J = so3_left_jacobian(phi)
        # Should be close to identity
        assert np.allclose(J, np.eye(3), atol=1e-6)

    def test_skew_symmetric(self):
        """Test skew-symmetric matrix properties"""
        v = np.array([1, 2, 3])
        S = skew_symmetric(v)

        # Check skew-symmetry: S^T = -S
        np.testing.assert_array_almost_equal(S.T, -S)

        # Check trace is zero
        np.testing.assert_almost_equal(np.trace(S), 0.0)

        # Check cross product equivalence: S @ w = v x w
        w = np.array([4, 5, 6])
        np.testing.assert_array_almost_equal(S @ w, np.cross(v, w))


class TestRotationConversions:
    """Tests for rotation matrix conversions"""

    def test_elementary_rotations_orthogonal(self):
        """Test that elementary rotations are orthogonal"""
        angle = np.pi / 4
        for rot_func in [rotx, roty, rotz]:
            R = rot_func(angle)
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
            np.testing.assert_almost_equal(np.linalg.det(R), 1.0)

    def test_rotx(self):
        """Test rotation around x-axis"""
        R = rotx(np.pi / 2)
        # Rotating [0, 1, 0] by 90° around x gives [0, 0, 1]
        v = np.array([0, 1, 0])
        result = R @ v
        expected = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_roty(self):
        """Test rotation around y-axis"""
        R = roty(np.pi / 2)
        # Rotating [0, 0, 1] by 90° around y gives [1, 0, 0]
        v = np.array([0, 0, 1])
        result = R @ v
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotz(self):
        """Test rotation around z-axis"""
        R = rotz(np.pi / 2)
        # Rotating [1, 0, 0] by 90° around z gives [0, 1, 0]
        v = np.array([1, 0, 0])
        result = R @ v
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_from_rpy_identity(self):
        """Test RPY to rotation matrix for zero angles"""
        R = from_rpy(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rpy_roundtrip(self):
        """Test conversion: angles -> matrix -> angles"""
        roll_in = 0.1
        pitch_in = 0.2
        yaw_in = 0.3

        R = from_rpy(roll_in, pitch_in, yaw_in)
        roll_out, pitch_out, yaw_out = to_rpy(R)

        np.testing.assert_almost_equal(roll_in, roll_out)
        np.testing.assert_almost_equal(pitch_in, pitch_out)
        np.testing.assert_almost_equal(yaw_in, yaw_out)

    def test_rpy_gimbal_lock_positive(self):
        """Test gimbal lock at pitch = π/2"""
        R = from_rpy(0.1, np.pi / 2, 0.3)
        roll, pitch, yaw = to_rpy(R)
        np.testing.assert_almost_equal(pitch, np.pi / 2)
        # At gimbal lock, yaw is set to 0
        np.testing.assert_almost_equal(yaw, 0.0)

    def test_rpy_gimbal_lock_negative(self):
        """Test gimbal lock at pitch = -π/2"""
        R = from_rpy(0.1, -np.pi / 2, 0.3)
        roll, pitch, yaw = to_rpy(R)
        np.testing.assert_almost_equal(pitch, -np.pi / 2)
        np.testing.assert_almost_equal(yaw, 0.0)


class TestNormalization:
    """Tests for rotation matrix normalization"""

    def test_normalize_identity(self):
        """Test normalizing identity matrix"""
        R = normalize_rot(np.eye(3))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_normalize_valid_rotation(self):
        """Test normalizing already-valid rotation"""
        R_in = from_rpy(0.1, 0.2, 0.3)
        R_out = normalize_rot(R_in)
        np.testing.assert_array_almost_equal(R_in, R_out, decimal=10)

    def test_normalize_corrupted_rotation(self):
        """Test normalizing numerically drifted rotation matrix"""
        # Start with valid rotation
        R = from_rpy(0.1, 0.2, 0.3)

        # Add small noise to corrupt it
        R_corrupt = R + 1e-5 * np.random.randn(3, 3)

        # Normalize
        R_normalized = normalize_rot(R_corrupt)

        # Check properties are restored
        np.testing.assert_array_almost_equal(
            R_normalized @ R_normalized.T, np.eye(3), decimal=10
        )
        np.testing.assert_almost_equal(np.linalg.det(R_normalized), 1.0, decimal=10)

    def test_normalize_preserves_proper_rotation(self):
        """Test that normalization preserves det(R) = +1"""
        R = from_rpy(0.5, 0.3, 0.7)
        R_normalized = normalize_rot(R)
        assert np.linalg.det(R_normalized) > 0  # Proper rotation


class TestUmeyamaAlignment:
    """Tests for Umeyama alignment algorithm"""

    def test_umeyama_identity(self):
        """Test alignment of identical point sets"""
        x = np.random.randn(3, 10)
        r, t, c = umeyama_alignment(x, x, with_scale=False)

        np.testing.assert_array_almost_equal(r, np.eye(3), decimal=8)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=8)
        np.testing.assert_almost_equal(c, 1.0, decimal=8)

    def test_umeyama_pure_translation(self):
        """Test alignment with pure translation"""
        x = np.random.randn(3, 20)
        translation = np.array([1.0, 2.0, 3.0])
        y = x + translation[:, np.newaxis]

        r, t, c = umeyama_alignment(x, y, with_scale=False)

        np.testing.assert_array_almost_equal(r, np.eye(3), decimal=8)
        np.testing.assert_array_almost_equal(t, translation, decimal=8)
        np.testing.assert_almost_equal(c, 1.0, decimal=8)

    def test_umeyama_pure_rotation(self):
        """Test alignment with pure rotation"""
        x = np.random.randn(3, 20)
        R_true = from_rpy(0.1, 0.2, 0.3)
        y = R_true @ x

        r, t, c = umeyama_alignment(x, y, with_scale=False)

        np.testing.assert_array_almost_equal(r, R_true, decimal=8)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=8)
        np.testing.assert_almost_equal(c, 1.0, decimal=8)

    def test_umeyama_rotation_translation(self):
        """Test alignment with rotation and translation"""
        x = np.random.randn(3, 30)
        R_true = from_rpy(0.3, 0.1, 0.5)
        t_true = np.array([1.5, -2.0, 3.5])
        y = R_true @ x + t_true[:, np.newaxis]

        r, t, c = umeyama_alignment(x, y, with_scale=False)

        np.testing.assert_array_almost_equal(r, R_true, decimal=6)
        np.testing.assert_array_almost_equal(t, t_true, decimal=6)
        np.testing.assert_almost_equal(c, 1.0, decimal=8)

    def test_umeyama_with_scale(self):
        """Test alignment with rotation, translation, and scale"""
        x = np.random.randn(3, 30)
        R_true = from_rpy(0.2, 0.3, 0.4)
        t_true = np.array([1.0, 2.0, 3.0])
        c_true = 2.5
        y = c_true * R_true @ x + t_true[:, np.newaxis]

        r, t, c = umeyama_alignment(x, y, with_scale=True)

        np.testing.assert_array_almost_equal(r, R_true, decimal=6)
        np.testing.assert_array_almost_equal(t, t_true, decimal=6)
        np.testing.assert_almost_equal(c, c_true, decimal=6)

    def test_umeyama_proper_rotation(self):
        """Test that Umeyama returns proper rotation (det = +1)"""
        x = np.random.randn(3, 20)
        y = np.random.randn(3, 20)

        r, t, c = umeyama_alignment(x, y, with_scale=False)

        # Check it's a proper rotation
        assert np.linalg.det(r) > 0
        np.testing.assert_array_almost_equal(r @ r.T, np.eye(3), decimal=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
