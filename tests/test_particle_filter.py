"""
Tests for Stage 3 (particle filter) and Stage 4 (transition model).

Tests:
  - LatentWorldModel.forward_batched produces correct shapes
  - LatentWorldModel.decode produces correct shapes
  - ParticleFilter init, run_chunk, weight update, resampling, weighted estimate
  - TransitionModel forward shapes and near-identity init
  - Integration: particle filter with transition model
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.torch_iekf import TorchIEKF
from src.models.world_model import LatentWorldModel
from src.models.transition_model import TransitionModel
from src.particle_filter import ParticleFilter


@pytest.fixture
def iekf():
    """Build a TorchIEKF with LatentWorldModel attached."""
    model = TorchIEKF()
    model.world_model = LatentWorldModel(
        input_channels=6,
        cnn_channels=32,
        kernel_size=5,
        cnn_dilation=3,
        cnn_dropout=0.1,
        latent_dim=8,
        measurement_decoder={"enabled": True, "beta": 3.0},
        process_decoder={"enabled": True, "alpha": 3.0},
        weight_scale=0.1,
        bias_scale=0.1,
    )
    model.set_Q()
    return model


@pytest.fixture
def imu_data():
    """Generate fake IMU data: (N, 6) raw, (1, 6, N) normalized."""
    N = 50
    u_raw = torch.randn(N, 6) * 0.1
    u_norm = u_raw.t().unsqueeze(0)  # (1, 6, N)
    return u_raw, u_norm, N


class TestWorldModelBatched:
    """Test LatentWorldModel.forward_batched."""

    def test_forward_batched_shapes(self, iekf, imu_data):
        u_raw, u_norm, N = imu_data
        M = 5
        wm = iekf.world_model
        wm.train()
        out = wm.forward_batched(u_norm, iekf, M)

        assert out.measurement_covs.shape == (M, N, 2)
        assert out.acc_bias_corrections.shape == (M, N, 3)
        assert out.gyro_bias_corrections.shape == (M, N, 3)
        assert out.bias_noise_scaling.shape == (M, N, 3)
        assert out.mu_z.shape == (N, 8)
        assert out.log_var_z.shape == (N, 8)

    def test_forward_batched_particles_differ_in_training(self, iekf, imu_data):
        """In training mode, M particles should have different z samples."""
        u_raw, u_norm, N = imu_data
        M = 5
        wm = iekf.world_model
        wm.train()
        out = wm.forward_batched(u_norm, iekf, M)

        # Measurement covs should differ across particles
        diffs = (out.measurement_covs[0] - out.measurement_covs[1]).abs().sum()
        assert diffs > 0, "Particles should differ in training mode"

    def test_forward_batched_eval_mode(self, iekf, imu_data):
        """In eval mode, all particles should use mu_z (identical)."""
        u_raw, u_norm, N = imu_data
        M = 5
        wm = iekf.world_model
        wm.eval()
        out = wm.forward_batched(u_norm, iekf, M)

        # All particles should be identical in eval mode
        for m in range(1, M):
            torch.testing.assert_close(
                out.measurement_covs[0], out.measurement_covs[m],
            )

    def test_decode_shapes(self, iekf):
        """Test decode-only path used by transition model."""
        wm = iekf.world_model
        M, latent_dim = 5, 8
        z = torch.randn(M, latent_dim)
        out = wm.decode(z, iekf)

        assert out.measurement_covs.shape == (M, 2)
        assert out.acc_bias_corrections.shape == (M, 3)
        assert out.gyro_bias_corrections.shape == (M, 3)
        assert out.bias_noise_scaling.shape == (M, 3)

    def test_decode_2d_shapes(self, iekf):
        """Test decode with (M, N, latent_dim) input."""
        wm = iekf.world_model
        M, N, latent_dim = 5, 20, 8
        z = torch.randn(M, N, latent_dim)
        out = wm.decode(z, iekf)

        assert out.measurement_covs.shape == (M, N, 2)
        assert out.acc_bias_corrections.shape == (M, N, 3)


class TestParticleFilter:
    """Test ParticleFilter mechanics."""

    def test_init_particles(self, iekf, imu_data):
        u_raw, _, N = imu_data
        M = 5
        t = torch.arange(N, dtype=torch.float32) * 0.01
        v_gt = torch.zeros(N, 3)
        ang0 = torch.zeros(3)

        single_state = iekf.init_state(t, u_raw, v_gt, ang0)
        pf = ParticleFilter(iekf, M)
        state, log_w = pf.init_particles(single_state)

        assert state["Rot"].shape == (M, 3, 3)
        assert state["v"].shape == (M, 3)
        assert state["P"].shape == (M, 21, 21)
        assert log_w.shape == (M,)
        # Uniform weights
        torch.testing.assert_close(
            torch.exp(log_w),
            torch.ones(M) / M,
            atol=1e-6, rtol=1e-6,
        )

    def test_update_weights(self, iekf):
        M = 5
        pf = ParticleFilter(iekf, M)

        # Uniform initial weights
        log_w = torch.full((M,), -torch.log(torch.tensor(float(M))))
        innovation = torch.randn(M, 2) * 0.01
        S = torch.eye(2).unsqueeze(0).expand(M, -1, -1) * 0.1

        new_log_w = pf.update_weights(innovation, S, log_w)
        assert new_log_w.shape == (M,)
        # Should sum to 1 in probability space
        w = torch.exp(new_log_w)
        torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_resample_if_needed_no_resample(self, iekf):
        """When ESS is high, no resampling should occur."""
        M = 5
        pf = ParticleFilter(iekf, M, resample_threshold=0.5)

        # Uniform weights → ESS = M, well above threshold
        log_w = torch.full((M,), -torch.log(torch.tensor(float(M))))
        state = {
            "Rot": torch.eye(3).unsqueeze(0).expand(M, -1, -1).clone(),
            "v": torch.zeros(M, 3),
            "p": torch.arange(M, dtype=torch.float32).unsqueeze(-1).expand(M, 3).clone(),
            "b_omega": torch.zeros(M, 3),
            "b_acc": torch.zeros(M, 3),
            "Rot_c_i": torch.eye(3).unsqueeze(0).expand(M, -1, -1).clone(),
            "t_c_i": torch.zeros(M, 3),
            "P": torch.eye(21).unsqueeze(0).expand(M, -1, -1).clone(),
        }
        new_state, new_log_w = pf.resample_if_needed(state, log_w)
        # No resampling → state unchanged
        torch.testing.assert_close(new_state["p"], state["p"])

    def test_resample_if_needed_triggers(self, iekf):
        """When one particle dominates, resampling should trigger."""
        M = 5
        pf = ParticleFilter(iekf, M, resample_threshold=0.5, jitter_std=0.0)

        # One dominant particle
        log_w = torch.tensor([-100., -100., -100., -100., 0.0])
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        state = {
            "Rot": torch.eye(3).unsqueeze(0).expand(M, -1, -1).clone(),
            "v": torch.zeros(M, 3),
            "p": torch.arange(M, dtype=torch.float32).unsqueeze(-1).expand(M, 3).clone(),
            "b_omega": torch.zeros(M, 3),
            "b_acc": torch.zeros(M, 3),
            "Rot_c_i": torch.eye(3).unsqueeze(0).expand(M, -1, -1).clone(),
            "t_c_i": torch.zeros(M, 3),
            "P": torch.eye(21).unsqueeze(0).expand(M, -1, -1).clone(),
        }
        new_state, new_log_w = pf.resample_if_needed(state, log_w)

        # After resampling, weights should be uniform
        expected_w = torch.ones(M) / M
        torch.testing.assert_close(
            torch.exp(new_log_w), expected_w, atol=1e-5, rtol=1e-5,
        )

    def test_weighted_estimate_shapes(self, iekf):
        M, K = 5, 20
        pf = ParticleFilter(iekf, M)

        traj = (
            torch.randn(M, K, 3, 3),  # Rot (not SO(3), but shape test)
            torch.randn(M, K, 3),      # v
            torch.randn(M, K, 3),      # p
            torch.randn(M, K, 3),      # b_omega
            torch.randn(M, K, 3),      # b_acc
            torch.randn(M, K, 3, 3),   # Rot_c_i
            torch.randn(M, K, 3),      # t_c_i
        )
        log_w = torch.full((M,), -torch.log(torch.tensor(float(M))))
        Rot_mean, p_mean = pf.weighted_estimate(traj, log_w)

        assert Rot_mean.shape == (K, 3, 3)
        assert p_mean.shape == (K, 3)

    def test_run_chunk_shapes(self, iekf, imu_data):
        """Test full run_chunk produces correct output shapes."""
        u_raw, u_norm, N = imu_data
        M = 3
        K = min(N, 30)
        t = torch.arange(K, dtype=torch.float32) * 0.01
        v_gt = torch.zeros(K, 3)
        ang0 = torch.zeros(3)

        iekf.world_model.train()
        single_state = iekf.init_state(t, u_raw[:K], v_gt, ang0)
        pf = ParticleFilter(iekf, M)
        state, log_w = pf.init_particles(single_state)

        # Get world model output
        u_norm_chunk = u_raw[:K].t().unsqueeze(0)
        wm_out = iekf.world_model.forward_batched(u_norm_chunk, iekf, M)

        traj, new_state, new_log_w, _ = pf.run_chunk(
            state, t, u_raw[:K], wm_out, log_w,
        )

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = traj
        assert Rot.shape == (M, K, 3, 3)
        assert p.shape == (M, K, 3)
        assert new_state["Rot"].shape == (M, 3, 3)
        assert new_log_w.shape == (M,)


class TestTransitionModel:
    """Test TransitionModel."""

    def test_forward_shapes(self):
        tm = TransitionModel(latent_dim=8, hidden_dim=64)
        tm.train()
        z = torch.randn(5, 8)
        imu = torch.randn(5, 6)
        z_next, mu, log_var = tm(z, imu)

        assert z_next.shape == (5, 8)
        assert mu.shape == (5, 8)
        assert log_var.shape == (5, 8)

    def test_near_zero_init(self):
        """With small init, output should be near zero."""
        tm = TransitionModel(latent_dim=8, hidden_dim=64, weight_scale=0.01)
        tm.eval()
        z = torch.zeros(1, 8)
        imu = torch.zeros(1, 6)
        z_next, mu, log_var = tm(z, imu)

        # Near-zero init → mu ≈ alpha * tanh(~0) ≈ 0
        assert mu.abs().max() < 0.1, f"mu should be near zero, got {mu}"

    def test_eval_deterministic(self):
        """In eval mode, same input should give same output."""
        tm = TransitionModel(latent_dim=8)
        tm.eval()
        z = torch.randn(3, 8)
        imu = torch.randn(3, 6)
        z1, _, _ = tm(z, imu)
        z2, _, _ = tm(z, imu)
        torch.testing.assert_close(z1, z2)

    def test_batched_forward(self):
        """Test with 2D batch input."""
        tm = TransitionModel(latent_dim=8)
        tm.train()
        M, N = 5, 20
        z = torch.randn(M, 8)
        imu = torch.randn(6)  # shared IMU
        # Expand imu for all particles
        imu_batch = imu.unsqueeze(0).expand(M, -1)
        z_next, mu, log_var = tm(z, imu_batch)

        assert z_next.shape == (M, 8)


class TestParticleFilterWithTransition:
    """Test particle filter with transition model integration."""

    def test_run_chunk_with_transition(self, iekf, imu_data):
        """Test run_chunk with transition model produces correct shapes."""
        u_raw, u_norm, N = imu_data
        M = 3
        K = min(N, 30)
        t = torch.arange(K, dtype=torch.float32) * 0.01
        v_gt = torch.zeros(K, 3)
        ang0 = torch.zeros(3)

        tm = TransitionModel(latent_dim=8, hidden_dim=64)
        tm.train()
        iekf.world_model.train()

        single_state = iekf.init_state(t, u_raw[:K], v_gt, ang0)
        pf = ParticleFilter(iekf, M)
        state, log_w = pf.init_particles(single_state)

        # Initialize z_particles
        z_particles = torch.randn(M, 8)

        traj, new_state, new_log_w, z_final = pf.run_chunk(
            state, t, u_raw[:K], None, log_w,
            z_particles=z_particles,
            transition_model=tm,
            world_model=iekf.world_model,
        )

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = traj
        assert Rot.shape == (M, K, 3, 3)
        assert p.shape == (M, K, 3)
        assert z_final.shape == (M, 8)
        assert new_log_w.shape == (M,)

    def test_transition_model_z_evolves(self, iekf, imu_data):
        """Z should change when transition model propagates."""
        u_raw, _, N = imu_data
        M = 3
        K = min(N, 20)
        t = torch.arange(K, dtype=torch.float32) * 0.01
        v_gt = torch.zeros(K, 3)
        ang0 = torch.zeros(3)

        tm = TransitionModel(latent_dim=8, hidden_dim=64, weight_scale=0.1)
        tm.train()
        iekf.world_model.train()

        single_state = iekf.init_state(t, u_raw[:K], v_gt, ang0)
        pf = ParticleFilter(iekf, M)
        state, log_w = pf.init_particles(single_state)

        z_init = torch.randn(M, 8)
        _, _, _, z_final = pf.run_chunk(
            state, t, u_raw[:K], None, log_w,
            z_particles=z_init.clone(),
            transition_model=tm,
            world_model=iekf.world_model,
        )

        # Z should have changed after K-1 transition steps
        diff = (z_final - z_init).abs().sum()
        assert diff > 0, "Z should evolve through transition model"
