"""
MeasurementCovNet (MesNet): CNN that predicts measurement covariance from raw IMU signals.

This network converts raw IMU signals (gyroscope + accelerometer, 6 channels)
into per-timestep measurement covariance matrices for the zero-velocity
constraints. It operates directly on the IMU signals without knowledge of
the filter's state estimates.

Architecture (default):
    Conv1d(6, 32, k=5) -> ReplicationPad -> ReLU -> Dropout(0.5) ->
    Conv1d(32, 32, k=5, dilation=3) -> ReplicationPad -> ReLU -> Dropout(0.5) ->
    Linear(32, 2) -> Tanh

Output: Per-timestep measurement covariances [lateral, vertical].
"""

import torch
import torch.nn as nn

from src.models.base_covariance_net import BaseCovarianceNet


class MeasurementCovNet(BaseCovarianceNet):
    """
    CNN-based measurement covariance prediction network.

    Processes raw IMU data through a 1D CNN to predict adaptive measurement
    noise covariances for each timestep. The output scales the baseline
    measurement covariance.

    Args:
        input_channels: Number of IMU input channels (default: 6, [gyro, acc])
        output_dim: Number of measurement covariance outputs (default: 2, [lateral, vertical])
        initial_beta: Initial value for beta scaling parameter (default: 3.0)
        cnn_channels: Number of channels in CNN hidden layers (default: 32)
        kernel_size: CNN kernel size (default: 5)
        dilation: Dilation for second conv layer (default: 3)
        dropout: Dropout probability (default: 0.5)
        bias_scale: Factor to scale initial linear layer biases (default: 0.01)
        weight_scale: Factor to scale initial linear layer weights (default: 0.01)
    """

    def __init__(
        self,
        input_channels=6,
        output_dim=2,
        initial_beta=3.0,
        cnn_channels=32,
        kernel_size=5,
        dilation=3,
        dropout=0.5,
        bias_scale=0.01,
        weight_scale=0.01,
    ):
        super().__init__()

        self.output_dim = output_dim

        # Beta scaling parameter (not learned, registered as buffer for device tracking)
        self.register_buffer(
            "beta_measurement",
            initial_beta * torch.ones(output_dim).double(),
        )

        self.tanh = nn.Tanh()

        # CNN backbone
        self.cov_net = nn.Sequential(
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
        ).double()

        # Linear head
        self.cov_lin = nn.Sequential(
            nn.Linear(cnn_channels, output_dim),
            nn.Tanh(),
        ).double()

        # Initialize linear layer weights to be small
        self.cov_lin[0].bias.data[:] *= bias_scale
        self.cov_lin[0].weight.data[:] *= weight_scale

    def forward(self, u, iekf):
        """
        Predict measurement covariances from normalized IMU input.

        Args:
            u: Normalized IMU input tensor (1, input_channels, seq_len)
            iekf: Reference to IEKF filter (provides cov0_measurement baseline)

        Returns:
            Measurement covariances (seq_len, output_dim)
        """
        # CNN processes temporal IMU signal
        y_cov = self.cov_net(u).transpose(0, 2).squeeze()

        # Linear head produces per-timestep scaling factors
        z_cov = self.cov_lin(y_cov)

        # Apply beta scaling
        z_cov_net = self.beta_measurement.unsqueeze(0) * z_cov

        # Scale baseline measurement covariance
        measurements_covs = iekf.cov0_measurement.unsqueeze(0) * (
            10**z_cov_net
        )

        return measurements_covs

    def get_output_dim(self):
        return self.output_dim
