"""
LearnedBiasCorrectionNet: causal 1D CNN that predicts per-timestep
accelerometer bias corrections from normalized IMU signals.

Instead of replacing the IEKF dynamics entirely (as NeuralODEDynamics
attempted), this network predicts an additive correction Δb_acc(t) to the
accelerometer bias at each timestep.  The classical SE(3) kinematics
remain intact:

    acc = R @ (a_imu - b_acc - Δb_acc(t)) + g
    v_next = v + acc * dt
    p_next = p + v * dt + 0.5 * acc * dt²

This is the right formulation because:
- Exact geometric integration is preserved (SO(3) rotation, gravity)
- The network only learns what physics cannot model: temporally-correlated
  systematic accelerometer errors (vibration, road effects, nonlinearities)
- A causal dilated 1D CNN captures these temporal patterns, unlike a
  per-step Markovian ODE which has no temporal context

Architecture (same backbone as MeasurementCovNet):
    Conv1d(6, 32, k=5) -> ReplicationPad -> ReLU -> Dropout(0.5) ->
    Conv1d(32, 32, k=5, dilation=3) -> ReplicationPad -> ReLU -> Dropout(0.5) ->
    Linear(32, 3)

Output: Per-timestep bias correction [Δb_acc_x, Δb_acc_y, Δb_acc_z].
"""

import torch
import torch.nn as nn

from src.models.base_covariance_net import BaseCovarianceNet


class LearnedBiasCorrectionNet(BaseCovarianceNet):
    """
    Causal 1D CNN that predicts accelerometer bias corrections.

    Processes normalized IMU data through a dilated causal CNN to produce
    per-timestep additive corrections to the accelerometer bias estimate.
    The output is bounded by tanh and scaled, so corrections cannot
    exceed a physically reasonable range.

    Args:
        input_channels: Number of IMU input channels (default: 6, [gyro, acc])
        output_dim: Number of correction outputs (default: 3, [Δb_x, Δb_y, Δb_z])
        max_correction: Maximum correction magnitude in m/s² (default: 0.5)
        cnn_channels: Number of channels in CNN hidden layers (default: 32)
        kernel_size: CNN kernel size (default: 5)
        dilation: Dilation for second conv layer (default: 3)
        dropout: Dropout probability (default: 0.5)
        weight_scale: Factor to scale initial weights (default: 0.01)
        bias_scale: Factor to scale initial biases (default: 0.01)
    """

    def __init__(
        self,
        input_channels=6,
        output_dim=3,
        max_correction=0.5,
        cnn_channels=32,
        kernel_size=5,
        dilation=3,
        dropout=0.5,
        weight_scale=0.01,
        bias_scale=0.01,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.max_correction = max_correction

        self.tanh = nn.Tanh()

        # CNN backbone — same proven pattern as MeasurementCovNet
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
        )

        # Linear head → tanh-bounded output
        self.correction_head = nn.Sequential(
            nn.Linear(cnn_channels, output_dim),
            nn.Tanh(),
        )

        # Near-zero initialization: network starts as identity (no correction)
        self.correction_head[0].weight.data[:] *= weight_scale
        self.correction_head[0].bias.data[:] *= bias_scale

    def forward(self, u, iekf=None):
        """
        Predict per-timestep accelerometer bias corrections.

        Args:
            u: Normalized IMU input tensor (1, input_channels, seq_len).
            iekf: Reference to IEKF filter (unused, kept for interface
                  compatibility with BaseCovarianceNet).

        Returns:
            Bias corrections (seq_len, 3) in m/s², bounded to
            [-max_correction, +max_correction].
        """
        # CNN extracts temporal features from IMU signal
        features = self.cov_net(u).transpose(0, 2).squeeze()

        # Linear head produces bounded correction per timestep
        correction = self.correction_head(features)

        # Scale to physical range
        return correction * self.max_correction

    def get_output_dim(self):
        return self.output_dim
