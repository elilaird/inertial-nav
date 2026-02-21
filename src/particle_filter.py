"""
Particle filter for the IEKF with latent context.

Manages M particles, each running a batched IEKF instance with its own
z-driven corrections. Handles log-weight accumulation, ESS-triggered
resampling with soft weights for gradient flow, and weighted state estimation.
"""

import torch
import torch.nn as nn


class ParticleFilter(nn.Module):
    """
    Manages M particles, each with a batched IEKF instance.

    Responsibilities:
    - Per-particle z sampling (delegates to world model forward_batched)
    - Batched IEKF propagation and update via TorchIEKF.run_chunk_batched
    - Log-weight accumulation and normalization
    - ESS-triggered resampling with soft weights for gradient flow
    - Weighted state estimate

    Args:
        iekf: TorchIEKF instance.
        M: Number of particles.
        resample_threshold: ESS fraction of M triggering resample.
        soft_alpha: Soft weight mixing coefficient for gradient flow.
        jitter_std: Post-resample state jitter std (applied to biases).
    """

    def __init__(self, iekf, M, resample_threshold=0.5, soft_alpha=0.5,
                 jitter_std=0.01):
        super().__init__()
        self.iekf = iekf
        self.M = M
        self.resample_threshold = resample_threshold
        self.soft_alpha = soft_alpha
        self.jitter_std = jitter_std

    def init_particles(self, single_state):
        """
        Replicate single-instance state for M particles.

        Args:
            single_state: Dict with (dim...) tensors from iekf.init_state().

        Returns:
            batched_state: Dict with (M, dim...) tensors.
            log_weights: (M,) uniform log weights.
        """
        M = self.M
        batched_state = {}
        for k, v in single_state.items():
            batched_state[k] = v.unsqueeze(0).expand(M, *v.shape).clone()
        log_weights = torch.full(
            (M,), -torch.log(torch.tensor(float(M))),
            device=single_state["Rot"].device,
            dtype=single_state["Rot"].dtype,
        )
        return batched_state, log_weights

    def run_chunk(self, state, t_chunk, u_chunk, wm_output, log_weights,
                   z_particles=None, transition_model=None, world_model=None):
        """
        Run particle filter for one BPTT chunk.

        Args:
            state: Dict with (M, ...) tensors.
            t_chunk: (K,) timestamps.
            u_chunk: (K, 6) shared IMU.
            wm_output: WorldModelOutput with (M, K, dim) tensors.
                When using transition model, this contains only initial
                chunk's encoder output; subsequent chunks use z_particles.
            log_weights: (M,) log weights from previous chunk.
            z_particles: (M, latent_dim) per-particle z values when using
                transition model. None for naive particle filter.
            transition_model: TransitionModel instance, or None.
            world_model: LatentWorldModel instance for decode-only calls
                when using transition model.

        Returns:
            traj: Tuple (Rot, v, p, ...) each (M, K, ...).
            new_state: Dict with (M, ...) tensors.
            log_weights: (M,) updated log weights.
            z_particles: (M, latent_dim) final z values if transition model
                is used, else None.
        """
        K = t_chunk.shape[0]
        M = state["Rot"].shape[0]
        dt_chunk = t_chunk[1:] - t_chunk[:-1]

        # Allocate trajectory tensors
        Rot = t_chunk.new_zeros(M, K, 3, 3)
        v_traj = t_chunk.new_zeros(M, K, 3)
        p_traj = t_chunk.new_zeros(M, K, 3)
        b_omega_traj = t_chunk.new_zeros(M, K, 3)
        b_acc_traj = t_chunk.new_zeros(M, K, 3)
        Rot_c_i_traj = t_chunk.new_zeros(M, K, 3, 3)
        t_c_i_traj = t_chunk.new_zeros(M, K, 3)

        # Seed first timestep
        Rot[:, 0] = state["Rot"]
        v_traj[:, 0] = state["v"]
        p_traj[:, 0] = state["p"]
        b_omega_traj[:, 0] = state["b_omega"]
        b_acc_traj[:, 0] = state["b_acc"]
        Rot_c_i_traj[:, 0] = state["Rot_c_i"]
        t_c_i_traj[:, 0] = state["t_c_i"]
        P = state["P"]

        use_transition = transition_model is not None and z_particles is not None

        for i in range(1, K):
            if use_transition:
                # Transition model: propagate z, then decode
                # u_chunk[i] is (6,), expand for M particles
                imu_i = u_chunk[i].unsqueeze(0).expand(M, -1)
                z_particles, _, _ = transition_model(z_particles, imu_i)
                wm_i = world_model.decode(z_particles, self.iekf)
                bc_i = wm_i.acc_bias_corrections if wm_i.acc_bias_corrections is not None else None
                gc_i = wm_i.gyro_bias_corrections if wm_i.gyro_bias_corrections is not None else None
                bns_i = wm_i.bias_noise_scaling if wm_i.bias_noise_scaling is not None else None
                mc_i = wm_i.measurement_covs if wm_i.measurement_covs is not None else None
            else:
                bc_i = wm_output.acc_bias_corrections[:, i] if wm_output.acc_bias_corrections is not None else None
                gc_i = wm_output.gyro_bias_corrections[:, i] if wm_output.gyro_bias_corrections is not None else None
                bns_i = wm_output.bias_noise_scaling[:, i] if wm_output.bias_noise_scaling is not None else None
                mc_i = wm_output.measurement_covs[:, i] if wm_output.measurement_covs is not None else None

            # Propagate
            (Rot_i, v_i, p_i, b_omega_i, b_acc_i,
             Rot_c_i_i, t_c_i_i, P_i) = self.iekf.propagate_batched(
                Rot[:, i - 1], v_traj[:, i - 1], p_traj[:, i - 1],
                b_omega_traj[:, i - 1], b_acc_traj[:, i - 1],
                Rot_c_i_traj[:, i - 1], t_c_i_traj[:, i - 1],
                P, u_chunk[i], dt_chunk[i - 1],
                bias_correction=bc_i,
                gyro_correction=gc_i,
                bias_noise_scaling=bns_i,
            )

            # Update
            (Rot_up, v_up, p_up, b_omega_up, b_acc_up,
             Rot_c_i_up, t_c_i_up, P_up) = self.iekf.update_batched(
                Rot_i, v_i, p_i, b_omega_i, b_acc_i,
                Rot_c_i_i, t_c_i_i, P_i,
                u_chunk[i], i, mc_i,
            )

            # Compute innovation and S for weight update
            innovation, S = self._compute_innovation_and_S(
                Rot_i, v_i, b_omega_i, Rot_c_i_i, t_c_i_i,
                P_i, u_chunk[i], mc_i,
            )
            log_weights = self.update_weights(innovation, S, log_weights)

            # Store
            Rot[:, i] = Rot_up
            v_traj[:, i] = v_up
            p_traj[:, i] = p_up
            b_omega_traj[:, i] = b_omega_up
            b_acc_traj[:, i] = b_acc_up
            Rot_c_i_traj[:, i] = Rot_c_i_up
            t_c_i_traj[:, i] = t_c_i_up
            P = P_up

            # Resample if needed
            state_i = dict(
                Rot=Rot[:, i], v=v_traj[:, i], p=p_traj[:, i],
                b_omega=b_omega_traj[:, i], b_acc=b_acc_traj[:, i],
                Rot_c_i=Rot_c_i_traj[:, i], t_c_i=t_c_i_traj[:, i],
                P=P,
            )
            state_i, log_weights = self.resample_if_needed(state_i, log_weights)
            # Write back resampled state
            Rot[:, i] = state_i["Rot"]
            v_traj[:, i] = state_i["v"]
            p_traj[:, i] = state_i["p"]
            b_omega_traj[:, i] = state_i["b_omega"]
            b_acc_traj[:, i] = state_i["b_acc"]
            Rot_c_i_traj[:, i] = state_i["Rot_c_i"]
            t_c_i_traj[:, i] = state_i["t_c_i"]
            P = state_i["P"]

        traj = (Rot, v_traj, p_traj, b_omega_traj, b_acc_traj,
                Rot_c_i_traj, t_c_i_traj)
        new_state = dict(
            Rot=Rot[:, -1], v=v_traj[:, -1], p=p_traj[:, -1],
            b_omega=b_omega_traj[:, -1], b_acc=b_acc_traj[:, -1],
            Rot_c_i=Rot_c_i_traj[:, -1], t_c_i=t_c_i_traj[:, -1],
            P=P,
        )
        return traj, new_state, log_weights, z_particles

    def _compute_innovation_and_S(self, Rot, v, b_omega, Rot_c_i, t_c_i,
                                   P, u, measurement_cov):
        """
        Compute innovation and innovation covariance S for weight update.

        This mirrors the first part of update_batched without applying the
        Kalman correction, extracting just the innovation r and S = H P H^T + R.

        Args:
            All inputs have leading dimension M (from the batched IEKF).

        Returns:
            innovation: (M, 2) zero-velocity innovation.
            S: (M, 2, 2) innovation covariance.
        """
        M = Rot.shape[0]

        # Body frame orientation
        Rot_body = torch.bmm(Rot, Rot_c_i)

        # Velocity in IMU frame
        v_imu = torch.bmm(Rot.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)

        # Angular velocity
        omega = u[:3].unsqueeze(0) - b_omega

        # Velocity in body frame
        v_imu_in_body = torch.bmm(
            Rot_c_i.transpose(1, 2), v_imu.unsqueeze(-1)
        ).squeeze(-1)
        skew_t = self.iekf.skew_torch_batched(t_c_i)
        v_body = v_imu_in_body + torch.bmm(skew_t, omega.unsqueeze(-1)).squeeze(-1)

        Omega = self.iekf.skew_torch_batched(omega)
        skew_v_imu = self.iekf.skew_torch_batched(v_imu)
        H_v_imu = torch.bmm(Rot_c_i.transpose(1, 2), skew_v_imu)

        # Build H
        H = P.new_zeros(M, 2, self.iekf.P_dim)
        H[:, :, 3:6] = Rot_body.transpose(1, 2)[:, 1:, :]
        H[:, :, 15:18] = H_v_imu[:, 1:, :]
        H[:, :, 9:12] = skew_t[:, 1:, :]
        H[:, :, 18:21] = -Omega[:, 1:, :]

        # Innovation
        innovation = -v_body[:, 1:]  # (M, 2)

        # S = H P H^T + R
        R = torch.diag_embed(measurement_cov)
        S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R

        return innovation, S

    def update_weights(self, innovation, S, log_weights):
        """
        Update particle log-weights using the Gaussian likelihood.

        log_w^i += -0.5 * (innovation^T @ S^{-1} @ innovation + log(det(S)))

        Args:
            innovation: (M, 2)
            S: (M, 2, 2)
            log_weights: (M,)

        Returns:
            Normalized log_weights: (M,)
        """
        # S^{-1} @ innovation  → (M, 2, 1)
        S_inv_innov = torch.linalg.solve(S, innovation.unsqueeze(-1))  # (M, 2, 1)
        # Mahalanobis: innovation^T @ S^{-1} @ innovation → (M,)
        mahal = torch.bmm(innovation.unsqueeze(1), S_inv_innov).squeeze(-1).squeeze(-1)
        # Log determinant of S → (M,)
        log_det_S = torch.logdet(S)

        log_likelihood = -0.5 * (mahal + log_det_S)
        log_weights = log_weights + log_likelihood

        # Normalize in log space
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)

        return log_weights

    def resample_if_needed(self, state, log_weights):
        """
        ESS-triggered resampling with soft weights for gradient flow.

        ESS = 1 / sum(w^2). Resample if ESS < M * resample_threshold.
        Soft resampling: w_soft = alpha * w + (1 - alpha) / M

        Args:
            state: Dict with (M, ...) tensors.
            log_weights: (M,)

        Returns:
            state: Possibly resampled state dict.
            log_weights: Reset to uniform if resampled, else unchanged.
        """
        M = log_weights.shape[0]
        weights = torch.exp(log_weights)  # (M,)

        # ESS
        ess = 1.0 / (weights ** 2).sum()

        if ess < M * self.resample_threshold:
            # Soft resampling weights for gradient flow
            w_soft = self.soft_alpha * weights + (1.0 - self.soft_alpha) / M

            # Multinomial resampling (detached indices)
            indices = torch.multinomial(w_soft, M, replacement=True)

            # Resample state
            new_state = {}
            for k, v in state.items():
                new_state[k] = v[indices].clone()

            # Add jitter to biases to prevent collapse
            if self.jitter_std > 0:
                for k in ["b_omega", "b_acc"]:
                    new_state[k] = new_state[k] + self.jitter_std * torch.randn_like(
                        new_state[k]
                    )

            # Reset weights to uniform
            log_weights = torch.full_like(
                log_weights, -torch.log(torch.tensor(float(M)))
            )

            return new_state, log_weights

        return state, log_weights

    def weighted_estimate(self, traj, log_weights):
        """
        Compute weighted mean trajectory from particle trajectories.

        For the RPE loss, we need a single (K, 3, 3) rotation trajectory
        and (K, 3) position trajectory. Positions and velocities are
        weighted means. Rotations are approximated via weighted mean
        (valid for small inter-particle spread).

        Args:
            traj: Tuple (Rot, v, p, ...) each (M, K, ...).
            log_weights: (M,) log weights.

        Returns:
            Rot_mean: (K, 3, 3) weighted mean rotation (re-orthogonalized).
            p_mean: (K, 3) weighted mean position.
        """
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = traj
        weights = torch.exp(log_weights)  # (M,)

        # (M,) → (M, 1, 1) for Rot, (M, 1) for vectors
        w_rot = weights.view(-1, 1, 1, 1)  # (M, 1, 1, 1)
        w_vec = weights.view(-1, 1, 1)     # (M, 1, 1)

        # Weighted mean position: (K, 3)
        p_mean = (w_vec * p).sum(dim=0)

        # Weighted mean rotation (approximate for small spread): (K, 3, 3)
        Rot_mean_raw = (w_rot * Rot).sum(dim=0)  # (K, 3, 3)

        # Project back to SO(3) via SVD
        U, _, Vh = torch.linalg.svd(Rot_mean_raw)
        Rot_mean = torch.bmm(U, Vh)
        # Fix determinant
        det = torch.det(Rot_mean)
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        Rot_mean = Rot_mean * sign

        return Rot_mean, p_mean
