"""
WorldModel: shared CNN+GRU backbone with independently toggleable heads.

Architecture:
    IMUFeatureExtractor (shared CNN) → GRU → {measurement, acc_bias, gyro_bias, Q} heads

The shared backbone extracts temporal features from normalized IMU data via a
causal dilated 1D CNN, then processes them through a GRU to obtain a latent
state z_t with memory of the full history.  Four optional heads read z_t:

    - measurement_head:   per-timestep R_t (measurement covariance)
    - acc_bias_head:      per-timestep Δb_acc (accelerometer bias correction)
    - gyro_bias_head:     per-timestep Δb_ω (gyroscope bias correction)
    - process_noise_head: Q scaling (per-timestep or per-chunk)

Each head can be enabled/disabled independently via config, allowing clean
A/B testing.  All heads use near-zero init so the model starts as identity.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.base_covariance_net import BaseCovarianceNet


@dataclass
class WorldModelOutput:
    """Container for world model predictions.  ``None`` = head disabled."""

    measurement_covs: Optional[torch.Tensor] = None
    """(N, 2) measurement covariance per timestep, or None."""

    acc_bias_corrections: Optional[torch.Tensor] = None
    """(N, 3) accelerometer bias correction per timestep, or None."""

    gyro_bias_corrections: Optional[torch.Tensor] = None
    """(N, 3) gyroscope bias correction per timestep, or None."""

    process_noise_scaling: Optional[torch.Tensor] = None
    """(N, 6) or (1, 6) process noise scaling factors, or None."""


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
            nn.Dropout(p=dropout),
            nn.Conv1d(
                cnn_channels, cnn_channels, kernel_size, dilation=dilation
            ),
            nn.ReplicationPad1d(kernel_size - 1),
            nn.ReLU(),
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


class WorldModel(BaseCovarianceNet):
    """
    Shared backbone with independently toggleable prediction heads.

    Constructor parameter names match the Hydra YAML config keys so that
    ``TorchIEKF._build_network`` can forward them without mapping.

    Args:
        input_channels:       IMU channels (default 6: gyro + acc).
        cnn_channels:         CNN hidden width.
        kernel_size:          CNN kernel size.
        cnn_dilation:         Dilation for second conv layer.
        cnn_dropout:          Dropout probability for CNN.
        gru_hidden_size:      GRU hidden-state dimension.
        gru_num_layers:       GRU layers (currently 1; accepted for config compat).
        gru_dropout:          GRU dropout (only used when ``gru_num_layers > 1``).
        measurement_cov_head: Head config dict (or None to disable).
        acc_bias_head:        Head config dict (or None to disable).
        gyro_bias_head:       Head config dict (or None to disable).
        process_noise_head:   Head config dict (or None to disable).
        weight_scale:         Near-zero init multiplier for head weights.
        bias_scale:           Near-zero init multiplier for head biases.
    """

    def __init__(
        self,
        input_channels: int = 6,
        cnn_channels: int = 32,
        kernel_size: int = 5,
        cnn_dilation: int = 3,
        cnn_dropout: float = 0.5,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
        measurement_cov_head: Optional[dict] = None,
        acc_bias_head: Optional[dict] = None,
        gyro_bias_head: Optional[dict] = None,
        process_noise_head: Optional[dict] = None,
        weight_scale: float = 0.01,
        bias_scale: float = 0.01,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden_size

        # ---- shared backbone ----
        self.feature_extractor = IMUFeatureExtractor(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size,
            dilation=cnn_dilation,
            dropout=cnn_dropout,
        )
        self.gru = nn.GRU(
            cnn_channels,
            gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ---- heads ----
        heads = {
            "measurement_cov": dict(measurement_cov_head or {}),
            "acc_bias": dict(acc_bias_head or {}),
            "gyro_bias": dict(gyro_bias_head or {}),
            "process_noise": dict(process_noise_head or {}),
        }
        self._build_heads(heads, gru_hidden_size, weight_scale, bias_scale)

    # ------------------------------------------------------------------ #
    # Head construction
    # ------------------------------------------------------------------ #

    def _build_heads(self, heads_cfg, hidden, ws, bs):
        """Instantiate each enabled head as a child module."""

        # -- measurement covariance R_t --
        mc = heads_cfg.get("measurement_cov", {})
        if mc.get("enabled", False):
            out_dim = mc.get("output_dim", 2)
            beta_val = mc.get("initial_beta", 3.0)
            self.measurement_head = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )
            self.register_buffer(
                "beta_measurement",
                beta_val * torch.ones(out_dim),
            )
            self._small_init(self.measurement_head[0], ws, bs)
        else:
            self.measurement_head = None

        # -- accelerometer bias correction Δb_acc --
        ab = heads_cfg.get("acc_bias", {})
        if ab.get("enabled", False):
            out_dim = ab.get("output_dim", 3)
            self.max_acc_correction = ab.get("max_correction", 0.5)
            self.acc_bias_head = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )
            self._small_init(self.acc_bias_head[0], ws, bs)
        else:
            self.acc_bias_head = None
            self.max_acc_correction = 0.0

        # -- gyroscope bias correction Δb_ω --
        gb = heads_cfg.get("gyro_bias", {})
        if gb.get("enabled", False):
            out_dim = gb.get("output_dim", 3)
            self.max_gyro_correction = gb.get("max_correction", 0.01)
            self.gyro_bias_head = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )
            self._small_init(self.gyro_bias_head[0], ws, bs)
        else:
            self.gyro_bias_head = None
            self.max_gyro_correction = 0.0

        # -- process noise scaling Q_t --
        pn = heads_cfg.get("process_noise", {})
        if pn.get("enabled", False):
            out_dim = pn.get("output_dim", 6)
            self.per_timestep_q = pn.get("per_timestep", False)
            self.process_noise_head = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )
            self._small_init(self.process_noise_head[0], ws, bs)
        else:
            self.process_noise_head = None
            self.per_timestep_q = False

    @staticmethod
    def _small_init(linear: nn.Linear, ws: float, bs: float):
        """Near-zero initialisation so the head starts as identity."""
        linear.weight.data.mul_(ws)
        linear.bias.data.mul_(bs)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, u: torch.Tensor, iekf=None) -> WorldModelOutput:
        """
        Run shared backbone + all enabled heads.

        Args:
            u:    (1, C, N) normalized IMU tensor.
            iekf: TorchIEKF instance (provides ``cov0_measurement`` baseline).

        Returns:
            ``WorldModelOutput`` dataclass.
        """
        # Shared feature extraction
        features = self.feature_extractor(u)  # (N, cnn_channels)
        z, _ = self.gru(features.unsqueeze(0))  # (1, N, gru_hidden)
        z = z.squeeze(0)  # (N, gru_hidden)

        out = WorldModelOutput()

        # -- measurement covariance --
        if self.measurement_head is not None and iekf is not None:
            mc_raw = self.measurement_head(z)  # (N, 2)
            mc_scaled = self.beta_measurement.unsqueeze(0) * mc_raw
            out.measurement_covs = iekf.cov0_measurement.unsqueeze(0) * (
                10**mc_scaled
            )

        # -- accelerometer bias correction --
        if self.acc_bias_head is not None:
            out.acc_bias_corrections = (
                self.acc_bias_head(z) * self.max_acc_correction
            )

        # -- gyroscope bias correction --
        if self.gyro_bias_head is not None:
            out.gyro_bias_corrections = (
                self.gyro_bias_head(z) * self.max_gyro_correction
            )

        # -- process noise scaling --
        if self.process_noise_head is not None:
            q_raw = self.process_noise_head(z)  # (N, 6)
            q_scaling = 10**q_raw  # range ~[0.1, 10]
            if self.per_timestep_q:
                out.process_noise_scaling = q_scaling
            else:
                # Per-chunk: mean-pool to a single (1, 6) vector
                out.process_noise_scaling = q_scaling.mean(dim=0, keepdim=True)

        return out

    def get_output_dim(self):
        """Not meaningful for multi-head model; returns GRU hidden dim."""
        return self.gru_hidden
