"""
Stage 2 verification: batched IEKF (M particles) matches single IEKF.

Tests that run_chunk_batched with M=1 produces bit-identical output
(within 1e-5 tolerance) to run_chunk on the same synthetic sequence.
Also tests M>1 for shape correctness and basic properties.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.torch_iekf import TorchIEKF


# ---- helpers ----

def make_synthetic_sequence(N=100, dt_val=0.01):
    """Create synthetic IMU data for a gently turning vehicle."""
    t = torch.arange(N, dtype=torch.float32) * dt_val
    u = torch.zeros(N, 6, dtype=torch.float32)
    u[:, 5] = 9.80655  # gravity-compensated accel
    # gentle yaw rate
    u[:, 2] = 0.05
    # small forward acceleration
    u[:, 3] = 0.2

    v_mes = torch.zeros(N, 3, dtype=torch.float32)
    v_mes[0] = torch.tensor([1.0, 0.0, 0.0])
    ang0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    meas_covs = torch.tensor(
        [[1.0, 10.0]], dtype=torch.float32
    ).expand(N, 2).clone()

    return t, u, v_mes, ang0, meas_covs


def make_corrections(N, M=None):
    """Create small random corrections for testing."""
    torch.manual_seed(42)
    if M is None:
        bc = 0.001 * torch.randn(N, 3)
        gc = 0.001 * torch.randn(N, 3)
        bns = torch.ones(N, 3) + 0.01 * torch.randn(N, 3)
        return bc, gc, bns
    else:
        bc = 0.001 * torch.randn(M, N, 3)
        gc = 0.001 * torch.randn(M, N, 3)
        bns = torch.ones(M, N, 3) + 0.01 * torch.randn(M, N, 3)
        return bc, gc, bns


# ==================================================================
# Batched geometry utilities
# ==================================================================

class TestBatchedGeometry:
    """Test batched geometry utilities match single-instance versions."""

    def test_skew_batched(self):
        M = 5
        x = torch.randn(M, 3)
        S_batch = TorchIEKF.skew_torch_batched(x)
        for i in range(M):
            S_single = TorchIEKF.skew_torch(x[i])
            assert torch.allclose(S_batch[i], S_single, atol=1e-7), \
                f"skew mismatch at index {i}"

    def test_so3exp_batched_matches_single(self):
        torch.manual_seed(0)
        M = 10
        phi = 0.3 * torch.randn(M, 3)
        R_batch = TorchIEKF.so3exp_torch_batched(phi)
        for i in range(M):
            R_single = TorchIEKF.so3exp_torch(phi[i])
            assert torch.allclose(R_batch[i], R_single, atol=1e-5), \
                f"so3exp mismatch at index {i}: max diff = {(R_batch[i] - R_single).abs().max()}"

    def test_so3exp_batched_small_angle(self):
        """Test near-zero angle branch."""
        phi = torch.tensor([[1e-12, 0, 0], [0, 1e-15, 0]], dtype=torch.float32)
        R = TorchIEKF.so3exp_torch_batched(phi)
        Id3 = torch.eye(3)
        for i in range(2):
            assert torch.allclose(R[i], Id3, atol=1e-5)

    def test_so3exp_batched_valid_rotation(self):
        torch.manual_seed(1)
        phi = torch.randn(8, 3)
        R = TorchIEKF.so3exp_torch_batched(phi)
        for i in range(8):
            assert torch.allclose(R[i] @ R[i].t(), torch.eye(3), atol=1e-5)
            assert torch.isclose(torch.det(R[i]), torch.tensor(1.0), atol=1e-5)

    def test_se23_exp_batched_matches_single(self):
        torch.manual_seed(2)
        M = 8
        xi = 0.1 * torch.randn(M, 9)
        R_batch, x_batch = TorchIEKF.se23_exp_torch_batched(xi)
        for i in range(M):
            R_single, x_single = TorchIEKF.se23_exp_torch(xi[i])
            assert torch.allclose(R_batch[i], R_single, atol=1e-5), \
                f"se23 Rot mismatch at {i}"
            assert torch.allclose(x_batch[i], x_single, atol=1e-5), \
                f"se23 x mismatch at {i}"

    def test_se23_exp_batched_small_angle(self):
        xi = torch.zeros(3, 9)
        R, x = TorchIEKF.se23_exp_torch_batched(xi)
        for i in range(3):
            assert torch.allclose(R[i], torch.eye(3), atol=1e-6)

    def test_normalize_rot_batched(self):
        torch.manual_seed(3)
        M = 5
        # Create noisy rotations
        phi = torch.randn(M, 3)
        R = TorchIEKF.so3exp_torch_batched(phi)
        R_noisy = R + 1e-4 * torch.randn(M, 3, 3)
        R_norm = TorchIEKF.normalize_rot_torch_batched(R_noisy)
        for i in range(M):
            assert torch.allclose(
                R_norm[i] @ R_norm[i].t(), torch.eye(3), atol=1e-4
            )


# ==================================================================
# M=1 batched vs single IEKF: the critical Stage 2 test
# ==================================================================

class TestBatchedMatchesSingle:
    """Verify M=1 batched output matches single-instance output exactly."""

    def test_propagate_batched_matches_single(self):
        """Single propagation step: batched M=1 vs single."""
        iekf = TorchIEKF()
        P = iekf.init_covariance()

        Rot = TorchIEKF.from_rpy_torch(
            torch.tensor(0.1), torch.tensor(0.05), torch.tensor(0.2)
        )
        v = torch.tensor([1.0, 0.05, -0.02])
        p = torch.tensor([5.0, 0.3, -0.1])
        b_omega = torch.tensor([0.001, -0.002, 0.0005])
        b_acc = torch.tensor([0.01, -0.005, 0.003])
        Rot_c_i = torch.eye(3)
        t_c_i = torch.tensor([0.0, 0.05, -0.1])
        u = torch.tensor([0.1, 0.0, 0.05, 0.2, 0.0, 9.8])
        dt = torch.tensor(0.01)

        bc = torch.tensor([0.001, -0.001, 0.0])
        gc = torch.tensor([0.002, 0.0, -0.001])
        bns = torch.tensor([1.1, 0.9, 1.05])

        # Single
        out_s = iekf.propagate(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, dt,
            bias_correction=bc, gyro_correction=gc, bias_noise_scaling=bns,
        )

        # Batched M=1
        out_b = iekf.propagate_batched(
            Rot.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0),
            b_omega.unsqueeze(0), b_acc.unsqueeze(0),
            Rot_c_i.unsqueeze(0), t_c_i.unsqueeze(0),
            P.unsqueeze(0), u, dt,
            bias_correction=bc.unsqueeze(0),
            gyro_correction=gc.unsqueeze(0),
            bias_noise_scaling=bns.unsqueeze(0),
        )

        labels = ["Rot", "v", "p", "b_omega", "b_acc", "Rot_c_i", "t_c_i", "P"]
        for name, s, b in zip(labels, out_s, out_b):
            assert torch.allclose(s, b.squeeze(0), atol=1e-5), \
                f"propagate mismatch in {name}: max diff = {(s - b.squeeze(0)).abs().max()}"

    def test_update_batched_matches_single(self):
        """Single update step: batched M=1 vs single."""
        iekf = TorchIEKF()
        P = iekf.init_covariance()

        Rot = TorchIEKF.from_rpy_torch(
            torch.tensor(0.1), torch.tensor(0.05), torch.tensor(0.2)
        )
        v = torch.tensor([1.0, 0.1, 0.05])
        p = torch.tensor([5.0, 0.3, -0.1])
        b_omega = torch.tensor([0.001, -0.002, 0.0005])
        b_acc = torch.tensor([0.01, -0.005, 0.003])
        Rot_c_i = torch.eye(3)
        t_c_i = torch.tensor([0.0, 0.05, -0.1])
        u = torch.tensor([0.05, 0.0, 0.02, 0.1, 0.0, 9.8])
        meas_cov = torch.tensor([1.0, 10.0])

        out_s = iekf.update(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, 1, meas_cov
        )
        out_b = iekf.update_batched(
            Rot.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0),
            b_omega.unsqueeze(0), b_acc.unsqueeze(0),
            Rot_c_i.unsqueeze(0), t_c_i.unsqueeze(0),
            P.unsqueeze(0), u, 1, meas_cov.unsqueeze(0),
        )

        labels = ["Rot", "v", "p", "b_omega", "b_acc", "Rot_c_i", "t_c_i", "P"]
        for name, s, b in zip(labels, out_s, out_b):
            assert torch.allclose(s, b.squeeze(0), atol=1e-5), \
                f"update mismatch in {name}: max diff = {(s - b.squeeze(0)).abs().max()}"

    def test_run_chunk_batched_M1_matches_single(self):
        """Full chunk: batched M=1 must match single run_chunk within 1e-5."""
        iekf = TorchIEKF()
        N = 50
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)
        bc, gc, bns = make_corrections(N)

        # Single
        state_s = iekf.init_state(t, u, v_mes, ang0)
        traj_s, new_state_s = iekf.run_chunk(
            state_s, t, u, meas_covs,
            bias_corrections_chunk=bc,
            gyro_corrections_chunk=gc,
            bias_noise_scaling_chunk=bns,
        )

        # Batched M=1
        state_b = iekf.init_state_batched(t, u, v_mes, ang0, M=1)
        traj_b, new_state_b = iekf.run_chunk_batched(
            state_b, t, u,
            meas_covs.unsqueeze(0),  # (1, N, 2)
            bias_corrections_chunk=bc.unsqueeze(0),
            gyro_corrections_chunk=gc.unsqueeze(0),
            bias_noise_scaling_chunk=bns.unsqueeze(0),
        )

        # Compare trajectories
        labels = ["Rot", "v", "p", "b_omega", "b_acc", "Rot_c_i", "t_c_i"]
        for name, s, b in zip(labels, traj_s, traj_b):
            diff = (s - b.squeeze(0)).abs().max().item()
            assert diff < 1e-5, \
                f"run_chunk traj mismatch in {name}: max diff = {diff}"

        # Compare final state
        for key in new_state_s:
            diff = (new_state_s[key] - new_state_b[key].squeeze(0)).abs().max().item()
            assert diff < 1e-5, \
                f"run_chunk final state mismatch in {key}: max diff = {diff}"

    def test_run_chunk_batched_M1_no_corrections(self):
        """Full chunk with no corrections: batched M=1 vs single."""
        iekf = TorchIEKF()
        N = 30
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)

        state_s = iekf.init_state(t, u, v_mes, ang0)
        traj_s, _ = iekf.run_chunk(state_s, t, u, meas_covs)

        state_b = iekf.init_state_batched(t, u, v_mes, ang0, M=1)
        traj_b, _ = iekf.run_chunk_batched(
            state_b, t, u, meas_covs.unsqueeze(0)
        )

        for name, s, b in zip(
            ["Rot", "v", "p", "b_omega", "b_acc", "Rot_c_i", "t_c_i"],
            traj_s, traj_b,
        ):
            diff = (s - b.squeeze(0)).abs().max().item()
            assert diff < 1e-5, \
                f"no-correction traj mismatch in {name}: max diff = {diff}"


# ==================================================================
# M > 1: shape and property tests
# ==================================================================

class TestBatchedMultiParticle:
    """Test batched IEKF with M > 1 particles."""

    def test_shapes_M5(self):
        """Verify output shapes for M=5."""
        iekf = TorchIEKF()
        M, N = 5, 20
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)

        state = iekf.init_state_batched(t, u, v_mes, ang0, M)
        assert state["Rot"].shape == (M, 3, 3)
        assert state["P"].shape == (M, 21, 21)

        # Broadcast meas_covs to (M, N, 2)
        mc_batched = meas_covs.unsqueeze(0).expand(M, -1, -1).clone()

        traj, new_state = iekf.run_chunk_batched(state, t, u, mc_batched)
        Rot, v_t, p_t, bo, ba, rci, tci = traj

        assert Rot.shape == (M, N, 3, 3)
        assert v_t.shape == (M, N, 3)
        assert p_t.shape == (M, N, 3)
        assert new_state["P"].shape == (M, 21, 21)

    def test_identical_particles_produce_identical_output(self):
        """All M particles start identical with same inputs → same output."""
        iekf = TorchIEKF()
        M, N = 4, 15
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)

        state = iekf.init_state_batched(t, u, v_mes, ang0, M)
        mc = meas_covs.unsqueeze(0).expand(M, -1, -1).clone()
        traj, _ = iekf.run_chunk_batched(state, t, u, mc)

        # All particles should produce identical trajectories
        for i in range(1, M):
            for j, name in enumerate(["Rot", "v", "p"]):
                diff = (traj[j][0] - traj[j][i]).abs().max().item()
                assert diff < 1e-6, \
                    f"Particle {i} differs from 0 in {name}: {diff}"

    def test_different_corrections_produce_different_output(self):
        """Particles with different corrections should diverge."""
        iekf = TorchIEKF()
        M, N = 3, 30
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)

        state = iekf.init_state_batched(t, u, v_mes, ang0, M)
        mc = meas_covs.unsqueeze(0).expand(M, -1, -1).clone()

        # Give each particle different bias corrections
        torch.manual_seed(99)
        bc = 0.01 * torch.randn(M, N, 3)

        traj, _ = iekf.run_chunk_batched(
            state, t, u, mc, bias_corrections_chunk=bc
        )

        # Particles should diverge
        p_traj = traj[2]  # (M, N, 3)
        max_spread = (p_traj[0, -1] - p_traj[1, -1]).abs().max().item()
        assert max_spread > 1e-4, \
            f"Particles with different corrections did not diverge: {max_spread}"

    def test_valid_rotations_M8(self):
        """All M=8 particles should maintain valid rotations throughout."""
        iekf = TorchIEKF()
        M, N = 8, 40
        t, u, v_mes, ang0, meas_covs = make_synthetic_sequence(N)

        state = iekf.init_state_batched(t, u, v_mes, ang0, M)
        mc = meas_covs.unsqueeze(0).expand(M, -1, -1).clone()
        traj, _ = iekf.run_chunk_batched(state, t, u, mc)

        Rot = traj[0]  # (M, N, 3, 3)
        Id3 = torch.eye(3)
        for m in range(M):
            for i in range(N):
                err = (Rot[m, i] @ Rot[m, i].t() - Id3).abs().max().item()
                assert err < 1e-4, \
                    f"Invalid rotation at particle {m}, step {i}: {err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
