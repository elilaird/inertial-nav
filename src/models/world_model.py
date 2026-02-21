"""
LatentWorldModel: probabilistic latent variable world model for the IEKF.

Architecture:
    IMUFeatureExtractor (shared CNN) → FC encoder head → (mu_z, log_var_z)
    Reparameterization: z = mu_z + exp(0.5 * log_var_z) * epsilon
    ProcessDecoder(z_sample)  → delta_b_omega, delta_b_a, Q_bias_scale
    MeasurementDecoder(mu_z)  → N_n (measurement covariance)

The two decoders use different z representations:

  - **Measurement decoder** receives deterministic ``mu_z``.  N_n is already
    a covariance, so stochasticity in z would create variance *in* a noise
    parameter with no natural home in the IEKF update.  The decoder learns
    to output large N_n when mu_z is near the prior (uncertain context),
    encoding the right epistemic behavior directly in its weights.

  - **Process decoder** receives a stochastic sample ``z_sample`` via the
    reparameterization trick.  The variance across samples propagates
    context uncertainty into ``Q_bias_scale``, inflating the bias noise
    covariance when the correction itself is uncertain.

At eval / deterministic mode (torch.no_grad): both decoders use mu_z.

The KL term KL(q(z|x) || N(0,I)) is returned via mu_z / log_var_z fields
in WorldModelOutput and added to the loss in the trainer.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.base_covariance_net import BaseCovarianceNet


@dataclass
class WorldModelOutput:
    """Container for world model predictions.  ``None`` = head/decoder disabled."""

    measurement_covs: Optional[torch.Tensor] = None
    """(N, 2) measurement covariance per timestep, or None."""

    acc_bias_corrections: Optional[torch.Tensor] = None
    """(N, 3) accelerometer bias correction per timestep, or None."""

    gyro_bias_corrections: Optional[torch.Tensor] = None
    """(N, 3) gyroscope bias correction per timestep, or None."""

    process_noise_scaling: Optional[torch.Tensor] = None
    """(N, 6) or (1, 6) process noise scaling factors, or None. Legacy field."""

    bias_noise_scaling: Optional[torch.Tensor] = None
    """(N, 3) per-axis Q_bias_scale applied to both b_omega and b_acc diagonals."""

    mu_z: Optional[torch.Tensor] = None
    """(N, latent_dim) posterior mean, used for KL loss."""

    log_var_z: Optional[torch.Tensor] = None
    """(N, latent_dim) posterior log-variance, used for KL loss."""


class IMUFeatureExtractor(nn.Module):
    """
    Shared causal dilated 1D CNN over normalized IMU signals.

    Same proven architecture as MeasurementCovNet / LearnedBiasCorrectionNet:
        Conv1d → ReplicationPad → ReLU → Dropout →
        Conv1d (dilated) → ReplicationPad → ReLU → Dropout

    Input:  (1, C, N) normalized IMU
    Output: (N, cnn_channels) per-timestep feature vectors
    """

    def __init__(
        self,
        input_channels: int = 6,
        cnn_channels: int = 32,
        kernel_size: int = 5,
        dilation: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size),
            nn.ReplicationPad1d(kernel_size - 1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                cnn_channels, cnn_channels, kernel_size, dilation=dilation
            ),
            nn.ReplicationPad1d(kernel_size - 1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(p=dropout),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (1, C, N) normalized IMU tensor.
        Returns:
            (N, cnn_channels) per-timestep features.
        """
        # (1, cnn_channels, N) → (N, cnn_channels)
        return self.net(u).transpose(1, 2).squeeze(0)


class ProcessDecoder(nn.Module):
    """
    MLP decoder: latent z → bias corrections + bias noise scaling.

    Architecture:
        FC: latent_dim → 32, ReLU
        FC: 32 → 9

    Output split:
        raw[:, 0:3] → delta_b_omega = sigma_bw * alpha * tanh(raw[:, 0:3])
        raw[:, 3:6] → delta_b_a     = sigma_ba * alpha * tanh(raw[:, 3:6])
        raw[:, 6:9] → Q_bias_scale  = exp(raw[:, 6:9])

    sigma_bw, sigma_ba are derived from the IEKF's Q at forward time.
    When the network outputs zero: delta_b = 0, Q_bias_scale = 1 (identity).
    """

    def __init__(
        self,
        latent_dim: int,
        alpha: float = 3.0,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 9),
        )
        self._small_init(weight_scale, bias_scale)

    def _small_init(self, ws: float, bs: float):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(ws)
                layer.bias.data.mul_(bs)

    def forward(
        self, z: torch.Tensor, sigma_bw: torch.Tensor, sigma_ba: torch.Tensor
    ) -> tuple:
        """
        Args:
            z:        (N, latent_dim)
            sigma_bw: scalar or (3,) – sqrt of gyro bias noise variance
            sigma_ba: scalar or (3,) – sqrt of acc bias noise variance
        Returns:
            delta_b_omega: (N, 3)
            delta_b_a:     (N, 3)
            Q_bias_scale:  (N, 3)
        """
        raw = self.net(z)  # (N, 9)
        delta_b_omega = sigma_bw * self.alpha * torch.tanh(raw[:, 0:3])
        delta_b_a = sigma_ba * self.alpha * torch.tanh(raw[:, 3:6])
        Q_bias_scale = torch.exp(raw[:, 6:9])
        return delta_b_omega, delta_b_a, Q_bias_scale


class MeasurementDecoder(nn.Module):
    """
    MLP decoder: latent z → measurement noise covariance N_n.

    Architecture:
        FC: latent_dim → 32, ReLU
        FC: 32 → 2

    Output:
        N_n = cov0_measurement * 10^(beta * tanh(raw))   per timestep

    When the network outputs zero: N_n = cov0_measurement (baseline).
    """

    def __init__(
        self,
        latent_dim: int,
        beta: float = 3.0,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.beta = beta
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 2),
        )
        self._small_init(weight_scale, bias_scale)

    def _small_init(self, ws: float, bs: float):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(ws)
                layer.bias.data.mul_(bs)

    def forward(
        self, z: torch.Tensor, cov0_measurement: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z:                (N, latent_dim)
            cov0_measurement: (2,) baseline measurement covariance
        Returns:
            (N, 2) measurement covariance per timestep
        """
        raw = self.net(z)  # (N, 2)
        return cov0_measurement.unsqueeze(0) * (
            10 ** (self.beta * torch.tanh(raw))
        )


class LatentWorldModel(BaseCovarianceNet):
    """
    Latent variable world model: probabilistic encoder + two MLP decoders.

    The encoder maps a window of IMU data to a distribution q(z|x) via a
    dilated CNN + FC head.  A single z sample (reparameterization trick) is
    passed to the ProcessDecoder and MeasurementDecoder.

    Constructor parameter names match the Hydra YAML config keys so that
    ``TorchIEKF._build_network`` can forward them without mapping.

    Args:
        input_channels:       IMU channels (default 6: gyro + acc).
        cnn_channels:         CNN hidden width.
        kernel_size:          CNN kernel size.
        cnn_dilation:         Dilation for second conv layer.
        cnn_dropout:          Dropout probability for CNN.
        latent_dim:           Dimension of latent variable z.
        measurement_decoder:  Decoder config dict (or None/disabled to skip).
        process_decoder:      Decoder config dict (or None/disabled to skip).
        weight_scale:         Near-zero init multiplier for all FC weights.
        bias_scale:           Near-zero init multiplier for all FC biases.
    """

    def __init__(
        self,
        input_channels: int = 6,
        cnn_channels: int = 32,
        kernel_size: int = 5,
        cnn_dilation: int = 3,
        cnn_dropout: float = 0.5,
        latent_dim: int = 8,
        measurement_decoder: Optional[dict] = None,
        process_decoder: Optional[dict] = None,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- shared CNN backbone ----
        self.feature_extractor = IMUFeatureExtractor(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size,
            dilation=cnn_dilation,
            dropout=cnn_dropout,
        )

        # ---- encoder FC head: cnn_channels → 2 * latent_dim (no activation) ----
        # self.encoder_fc = nn.Linear(cnn_channels, 2 * latent_dim)
        self.encoder_fc = nn.Sequential(
            nn.Linear(cnn_channels, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim),
        )

        # ---- decoders ----
        meas_cfg = dict(measurement_decoder or {})
        self.measurement_dec: Optional[MeasurementDecoder] = None
        if meas_cfg.get("enabled", False):
            self.measurement_dec = MeasurementDecoder(
                latent_dim=latent_dim,
                beta=meas_cfg.get("beta", 3.0),
                weight_scale=weight_scale,
                bias_scale=bias_scale,
            )

        proc_cfg = dict(process_decoder or {})
        self.process_dec: Optional[ProcessDecoder] = None
        if proc_cfg.get("enabled", False):
            self.process_dec = ProcessDecoder(
                latent_dim=latent_dim,
                alpha=proc_cfg.get("alpha", 3.0),
                weight_scale=weight_scale,
                bias_scale=bias_scale,
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, u: torch.Tensor, iekf=None) -> WorldModelOutput:
        """
        Run encoder + reparameterization + decoders.

        Args:
            u:    (1, C, N) normalized IMU tensor.
            iekf: TorchIEKF instance (provides ``cov0_measurement`` and ``Q``).

        Returns:
            ``WorldModelOutput`` dataclass.
        """
        # CNN features: (N, cnn_channels)
        features = self.feature_extractor(u)

        # Encoder FC: (N, 2 * latent_dim)
        enc_out = self.encoder_fc(features)
        mu_z = enc_out[:, : self.latent_dim]
        log_var_z = enc_out[:, self.latent_dim :]

        # Reparameterization: stochastic sample for process decoder only.
        # Measurement decoder always uses deterministic mu_z (see module docstring).
        if self.training:
            epsilon = torch.randn_like(mu_z)
            z_sample = mu_z + torch.exp(0.5 * log_var_z) * epsilon
        else:
            z_sample = mu_z

        out = WorldModelOutput(mu_z=mu_z, log_var_z=log_var_z)

        # ---- measurement decoder (deterministic mu_z) ----
        if self.measurement_dec is not None and iekf is not None:
            out.measurement_covs = self.measurement_dec(
                mu_z, iekf.cov0_measurement
            )

        # ---- process decoder (stochastic z_sample) ----
        if self.process_dec is not None and iekf is not None:
            # Derive sigma_bw and sigma_ba from the IEKF's Q matrix
            sigma_bw = iekf.Q[9, 9].sqrt()
            sigma_ba = iekf.Q[12, 12].sqrt()
            delta_b_omega, delta_b_a, Q_bias_scale = self.process_dec(
                z_sample, sigma_bw, sigma_ba
            )
            out.gyro_bias_corrections = delta_b_omega
            out.acc_bias_corrections = delta_b_a
            out.bias_noise_scaling = Q_bias_scale

        return out

    def forward_batched(self, u: torch.Tensor, iekf, M: int) -> WorldModelOutput:
        """
        Encoder runs once, sample M z trajectories, decode in batch.

        For the particle filter: each particle gets its own stochastic z sample
        from the shared encoder posterior. Both decoders use stochastic z per
        particle (unlike single-instance forward where measurement decoder uses
        deterministic mu_z).

        Args:
            u:    (1, C, N) normalized IMU tensor.
            iekf: TorchIEKF instance (provides ``cov0_measurement`` and ``Q``).
            M:    Number of particles.

        Returns:
            ``WorldModelOutput`` with (M, N, dim) shaped tensors for per-particle
            outputs, and (N, latent_dim) shaped mu_z/log_var_z for shared KL.
        """
        features = self.feature_extractor(u)          # (N, 32)
        enc_out = self.encoder_fc(features)            # (N, 2*latent_dim)
        mu_z = enc_out[:, :self.latent_dim]             # (N, 8)
        log_var_z = enc_out[:, self.latent_dim:]        # (N, 8)
        sigma_z = torch.exp(0.5 * log_var_z)           # (N, 8)

        N = mu_z.shape[0]

        # Sample M z trajectories from shared posterior
        if self.training:
            epsilon = torch.randn(M, N, self.latent_dim, device=mu_z.device)
            z_samples = mu_z.unsqueeze(0) + sigma_z.unsqueeze(0) * epsilon  # (M, N, 8)
        else:
            z_samples = mu_z.unsqueeze(0).expand(M, -1, -1)  # (M, N, 8)

        # Flatten for batched decoder forward: (M*N, latent_dim)
        z_flat = z_samples.reshape(-1, self.latent_dim)

        out = WorldModelOutput(mu_z=mu_z, log_var_z=log_var_z)

        # Measurement decoder: stochastic z per particle
        if self.measurement_dec is not None and iekf is not None:
            meas_flat = self.measurement_dec(z_flat, iekf.cov0_measurement)  # (M*N, 2)
            out.measurement_covs = meas_flat.view(M, N, 2)

        # Process decoder: stochastic z per particle
        if self.process_dec is not None and iekf is not None:
            sigma_bw = iekf.Q[9, 9].sqrt()
            sigma_ba = iekf.Q[12, 12].sqrt()
            delta_b_omega, delta_b_a, Q_bias_scale = self.process_dec(
                z_flat, sigma_bw, sigma_ba
            )
            out.gyro_bias_corrections = delta_b_omega.view(M, N, 3)
            out.acc_bias_corrections = delta_b_a.view(M, N, 3)
            out.bias_noise_scaling = Q_bias_scale.view(M, N, 3)

        return out

    def decode(self, z: torch.Tensor, iekf) -> WorldModelOutput:
        """
        Decode from pre-computed z values (no encoder call).

        Used by the transition model path: z values come from the transition
        model rather than the encoder.

        Args:
            z:    (..., latent_dim) latent values (any batch shape).
            iekf: TorchIEKF instance.

        Returns:
            WorldModelOutput with decoded fields matching z's batch shape.
        """
        orig_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.latent_dim)

        out = WorldModelOutput()

        if self.measurement_dec is not None and iekf is not None:
            meas = self.measurement_dec(z_flat, iekf.cov0_measurement)
            out.measurement_covs = meas.view(*orig_shape, 2)

        if self.process_dec is not None and iekf is not None:
            sigma_bw = iekf.Q[9, 9].sqrt()
            sigma_ba = iekf.Q[12, 12].sqrt()
            delta_b_omega, delta_b_a, Q_bias_scale = self.process_dec(
                z_flat, sigma_bw, sigma_ba
            )
            out.gyro_bias_corrections = delta_b_omega.view(*orig_shape, 3)
            out.acc_bias_corrections = delta_b_a.view(*orig_shape, 3)
            out.bias_noise_scaling = Q_bias_scale.view(*orig_shape, 3)

        return out

    def get_output_dim(self):
        """Returns latent dimension."""
        return self.latent_dim
