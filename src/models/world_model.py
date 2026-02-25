"""
World models: probabilistic latent variable world models for the IEKF.

LatentWorldModel (single-branch):
    IMUFeatureExtractor (CNN) → FC encoder → mu_z → decoders
    Single latent z with ~17 timestep receptive field.

DualBranchWorldModel (dual-branch):
    LOCAL:  IMUFeatureExtractor (CNN, ~17 frames) → mu_local (deterministic) → MeasurementDecoder
    GLOBAL: LSTM (full causal sequence) → mu_global + learnable scalar sigma → ProcessDecoder
    Local branch is deterministic (no KL); regularized by L2 weight decay.
    Global branch uses learnable scalar log_sigma_global (init -1, sigma≈0.37) + KL.
    mu_z / log_var_z contain global branch only for the trainer's KL loss.

Decoder input conventions:
  - MeasurementDecoder receives deterministic mu_local.
  - ProcessDecoder receives z_global_sample (or cat(mu_local, z_global) in "concat" mode).
  - At eval: ProcessDecoder uses mu_global (no sampling).
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

    mu_local: Optional[torch.Tensor] = None
    """(N, local_latent_dim) deterministic local encoder output (DualBranchWorldModel only)."""


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
    MLP decoder: latent z → bias corrections + optional bias noise scaling.

    Two independent heads prevent gradient cross-contamination:

    bias_net:  FC(latent_dim → 32) → ReLU → LayerNorm → FC(32 → 6)
        raw[:, 0:3] → delta_b_omega = sigma_bw * alpha * tanh(raw[:, 0:3])
        raw[:, 3:6] → delta_b_a     = sigma_ba * alpha * tanh(raw[:, 3:6])

    q_net (optional):  FC(latent_dim → 32) → ReLU → LayerNorm → FC(32 → 3)
        raw → Q_bias_scale = exp(clamp(raw, -q_scale_clamp, q_scale_clamp))

    When use_bias_noise_scaling=False, q_net is not created and Q_bias_scale
    is always ones (no Q perturbation). Set this to False initially; enable
    once bias corrections are stable.

    sigma_bw, sigma_ba are derived from the IEKF's Q at forward time.
    When networks output zero: delta_b = 0, Q_bias_scale = 1 (identity).
    """

    def __init__(
        self,
        latent_dim: int,
        alpha: float = 3.0,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
        use_bias_noise_scaling: bool = True,
        q_scale_clamp: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.use_bias_noise_scaling = use_bias_noise_scaling
        self.q_scale_clamp = q_scale_clamp

        # Fix 2: separate head for bias corrections — isolated from Q_bias_scale
        self.bias_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 6),
        )

        # Fix 2 + 3: separate head for Q_bias_scale, only created when enabled
        self.q_net: Optional[nn.Sequential] = None
        if use_bias_noise_scaling:
            self.q_net = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
                nn.Linear(32, 3),
            )

        self._small_init(weight_scale, bias_scale)

    def _small_init(self, ws: float, bs: float):
        for net in [self.bias_net, self.q_net]:
            if net is None:
                continue
            for layer in net:
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
            Q_bias_scale:  (N, 3) — ones if use_bias_noise_scaling=False
        """
        raw_bias = self.bias_net(z)  # (N, 6)
        delta_b_omega = sigma_bw * self.alpha * torch.tanh(raw_bias[:, 0:3])
        delta_b_a = sigma_ba * self.alpha * torch.tanh(raw_bias[:, 3:6])

        if self.q_net is not None:
            raw_q = self.q_net(z)  # (N, 3)
            # Fix 1: clamp before exp — bounded to [exp(-c), exp(c)]
            Q_bias_scale = torch.exp(
                torch.clamp(raw_q, min=-self.q_scale_clamp, max=self.q_scale_clamp)
            )
        else:
            # Fix 3: no Q perturbation
            Q_bias_scale = torch.ones(
                z.shape[0], 3, dtype=z.dtype, device=z.device
            )

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

        # ---- encoder FC head: cnn_channels → latent_dim (mu_z only) ----
        self.encoder_fc = nn.Sequential(
            nn.Linear(cnn_channels, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

        # ---- single learnable noise scale for reparameterization ----
        # Initialised at 0 → sigma = exp(0) = 1; KL well-defined from the start.
        self.log_sigma = nn.Parameter(torch.zeros(1))

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
                use_bias_noise_scaling=proc_cfg.get("use_bias_noise_scaling", True),
                q_scale_clamp=proc_cfg.get("q_scale_clamp", 2.0),
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

        # Encoder FC: (N, latent_dim) — mu_z only
        mu_z = self.encoder_fc(features)

        # log_var_z broadcast from scalar: 2 * log_sigma gives the per-element
        # log-variance used by the KL loss in the trainer.
        log_var_z = self.log_sigma.mul(2).expand_as(mu_z)

        # Reparameterization: stochastic sample for process decoder only.
        # Measurement decoder always uses deterministic mu_z (see module docstring).
        if self.training:
            z_sample = mu_z + torch.exp(self.log_sigma) * torch.randn_like(mu_z)
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

    def get_output_dim(self):
        """Returns latent dimension."""
        return self.latent_dim


class DualBranchWorldModel(BaseCovarianceNet):
    """
    Dual-branch latent world model: separate local (CNN) and global (LSTM) encoders.

    The local branch uses a dilated CNN (~17 timestep receptive field) to encode
    instantaneous motion dynamics for the MeasurementDecoder.  The global branch
    uses an LSTM to accumulate full causal sequence context for the ProcessDecoder,
    capturing slowly-varying bias drift and environment state.

    Two independent VAE latent variables (z_local, z_global) each with their own
    learnable scalar log_sigma.  KL tensors are concatenated so the trainer's
    existing element-wise KL formula works unchanged.

    For BPTT training, call ``reset_hidden()`` at sequence start and
    ``detach_hidden()`` at chunk boundaries to implement truncated BPTT.

    Args:
        input_channels:         IMU channels (default 6: gyro + acc).
        local_cnn_channels:     CNN hidden width for local branch.
        local_kernel_size:      CNN kernel size for local branch.
        local_cnn_dilation:     Dilation for second conv layer.
        local_cnn_dropout:      Dropout probability for local CNN.
        local_latent_dim:       Dimension of local latent z_local.
        global_lstm_hidden:     LSTM hidden size for global branch.
        global_lstm_layers:     Number of LSTM layers.
        global_lstm_dropout:    Dropout between LSTM layers (0 if layers=1).
        global_latent_dim:      Dimension of global latent z_global.
        process_decoder_input:  "global" uses z_global only; "concat" uses
                                cat(z_local, z_global) for the ProcessDecoder.
        measurement_decoder:    Decoder config dict (or None/disabled to skip).
        process_decoder:        Decoder config dict (or None/disabled to skip).
        weight_scale:           Near-zero init multiplier for FC weights.
        bias_scale:             Near-zero init multiplier for FC biases.
    """

    def __init__(
        self,
        input_channels: int = 6,
        # Local branch (CNN)
        local_cnn_channels: int = 32,
        local_kernel_size: int = 5,
        local_cnn_dilation: int = 3,
        local_cnn_dropout: float = 0.5,
        local_latent_dim: int = 8,
        # Global branch (LSTM)
        global_lstm_hidden: int = 32,
        global_lstm_layers: int = 1,
        global_lstm_dropout: float = 0.0,
        global_latent_dim: int = 8,
        # Decoder z input mode
        process_decoder_input: str = "global",
        # Decoders
        measurement_decoder: Optional[dict] = None,
        process_decoder: Optional[dict] = None,
        # Init scales
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.local_latent_dim = local_latent_dim
        self.global_latent_dim = global_latent_dim
        self.process_decoder_input = process_decoder_input

        # ---- LOCAL BRANCH: CNN backbone ----
        self.feature_extractor = IMUFeatureExtractor(
            input_channels=input_channels,
            cnn_channels=local_cnn_channels,
            kernel_size=local_kernel_size,
            dilation=local_cnn_dilation,
            dropout=local_cnn_dropout,
        )

        self.local_encoder_fc = nn.Sequential(
            nn.Linear(local_cnn_channels, 2 * local_latent_dim),
            nn.ReLU(),
            nn.Linear(2 * local_latent_dim, local_latent_dim),
        )

        # ---- GLOBAL BRANCH: LSTM ----
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=global_lstm_hidden,
            num_layers=global_lstm_layers,
            dropout=global_lstm_dropout if global_lstm_layers > 1 else 0.0,
            batch_first=False,
        )

        self.global_encoder_fc = nn.Sequential(
            nn.Linear(global_lstm_hidden, 2 * global_latent_dim),
            nn.ReLU(),
            nn.Linear(2 * global_latent_dim, global_latent_dim),
        )

        self.log_sigma_global = nn.Parameter(torch.full((1,), -1.0))

        # LSTM hidden state for BPTT continuity
        self._lstm_hidden = None

        # ---- DECODERS ----
        # ProcessDecoder input dimension depends on mode
        if process_decoder_input == "concat":
            proc_latent_dim = local_latent_dim + global_latent_dim
        else:
            proc_latent_dim = global_latent_dim

        meas_cfg = dict(measurement_decoder or {})
        self.measurement_dec: Optional[MeasurementDecoder] = None
        if meas_cfg.get("enabled", False):
            self.measurement_dec = MeasurementDecoder(
                latent_dim=local_latent_dim,
                beta=meas_cfg.get("beta", 3.0),
                weight_scale=weight_scale,
                bias_scale=bias_scale,
            )

        proc_cfg = dict(process_decoder or {})
        self.process_dec: Optional[ProcessDecoder] = None
        if proc_cfg.get("enabled", False):
            self.process_dec = ProcessDecoder(
                latent_dim=proc_latent_dim,
                alpha=proc_cfg.get("alpha", 3.0),
                weight_scale=weight_scale,
                bias_scale=bias_scale,
                use_bias_noise_scaling=proc_cfg.get("use_bias_noise_scaling", True),
                q_scale_clamp=proc_cfg.get("q_scale_clamp", 2.0),
            )

    # ------------------------------------------------------------------ #
    # LSTM state management (for BPTT)
    # ------------------------------------------------------------------ #

    def reset_hidden(self):
        """Zero LSTM hidden state. Call at the start of each new sequence."""
        self._lstm_hidden = None

    def detach_hidden(self):
        """Detach LSTM hidden state from graph. Call at BPTT chunk boundaries."""
        if self._lstm_hidden is not None:
            self._lstm_hidden = tuple(h.detach() for h in self._lstm_hidden)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, u: torch.Tensor, iekf=None) -> WorldModelOutput:
        """
        Run dual-branch encoder + reparameterization + decoders.

        Args:
            u:    (1, C, N) normalized IMU tensor.
            iekf: TorchIEKF instance (provides ``cov0_measurement`` and ``Q``).

        Returns:
            ``WorldModelOutput`` dataclass with concatenated mu_z / log_var_z.
        """
        N = u.shape[2]

        # ---- LOCAL BRANCH (CNN) ----
        features = self.feature_extractor(u)  # (N, cnn_channels)
        mu_local = self.local_encoder_fc(features)  # (N, local_latent_dim)

        # Local branch is deterministic (no reparameterization); L2 via weight decay
        z_local = mu_local

        # ---- GLOBAL BRANCH (LSTM) ----
        # (1, C, N) → (N, 1, C) for LSTM (seq_len, batch, features)
        lstm_input = u.squeeze(0).t().unsqueeze(1)
        lstm_out, self._lstm_hidden = self.lstm(lstm_input, self._lstm_hidden)
        # (N, 1, hidden) → (N, hidden)
        lstm_features = lstm_out.squeeze(1)

        mu_global = self.global_encoder_fc(lstm_features)  # (N, global_latent_dim)

        if self.training:
            z_global = mu_global + torch.exp(
                self.log_sigma_global
            ) * torch.randn_like(mu_global)
        else:
            z_global = mu_global

        # ---- KL tensors (global branch only; local is deterministic) ----
        mu_z = mu_global
        log_var_z = self.log_sigma_global.mul(2).expand(N, self.global_latent_dim)

        out = WorldModelOutput(mu_z=mu_z, log_var_z=log_var_z, mu_local=mu_local)

        # ---- measurement decoder (deterministic mu_local) ----
        if self.measurement_dec is not None and iekf is not None:
            out.measurement_covs = self.measurement_dec(
                mu_local, iekf.cov0_measurement
            )

        # ---- process decoder (stochastic z_global or concat) ----
        if self.process_dec is not None and iekf is not None:
            if self.process_decoder_input == "concat":
                z_proc = torch.cat([z_local, z_global], dim=1)
            else:
                z_proc = z_global

            sigma_bw = iekf.Q[9, 9].sqrt()
            sigma_ba = iekf.Q[12, 12].sqrt()
            delta_b_omega, delta_b_a, Q_bias_scale = self.process_dec(
                z_proc, sigma_bw, sigma_ba
            )
            out.gyro_bias_corrections = delta_b_omega
            out.acc_bias_corrections = delta_b_a
            out.bias_noise_scaling = Q_bias_scale

        return out

    def get_output_dim(self):
        """Returns global latent dimension (mu_z contains global branch only)."""
        return self.global_latent_dim
