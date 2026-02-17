"""
InitProcessCovNet: learns initial state and process noise covariance scaling factors.

This network outputs multiplicative scaling factors for the IEKF's initial
state covariance and process noise covariance. It uses learnable linear layers
passed through tanh to produce bounded log-scale adjustments.

Output: 6 scaling factors each for initial and process covariances:
    [Rot, v, b_omega, b_acc, Rot_c_i, t_c_i]
"""

import torch
import torch.nn as nn

from src.models.base_covariance_net import BaseCovarianceNet


class InitProcessCovNet(BaseCovarianceNet):
    """
    Network that learns scaling factors for initial and process covariances.

    Architecture:
        - Linear(1, output_dim) -> tanh -> 10^x for initial covariance
        - Linear(1, output_dim) -> tanh -> 10^x for process covariance

    The tanh bounds the exponent to [-1, 1], so the scaling factors
    range from 0.1 to 10.0 (i.e., one order of magnitude adjustment).

    Args:
        output_dim: Number of covariance parameters to predict (default: 6)
        initial_beta: Initial value for beta parameters (default: 3.0)
        weight_scale: Factor to divide initial weights by (default: 10.0)
    """

    def __init__(self, output_dim=6, initial_beta=3.0, weight_scale=10.0):
        super().__init__()

        self.output_dim = output_dim

        # Beta parameters (not learned, registered as buffers for device tracking)
        self.register_buffer(
            "beta_process",
            initial_beta * torch.ones(2).double(),
        )
        self.register_buffer(
            "beta_initialization",
            initial_beta * torch.ones(2).double(),
        )

        # Learnable initial covariance scaling
        self.factor_initial_covariance = nn.Linear(
            1, output_dim, bias=False
        ).double()
        self.factor_initial_covariance.weight.data[:] /= weight_scale

        # Learnable process covariance scaling
        self.factor_process_covariance = nn.Linear(
            1, output_dim, bias=False
        ).double()
        self.factor_process_covariance.weight.data[:] /= weight_scale

        self.tanh = nn.Tanh()

    def forward(self, u, iekf):
        """Forward pass (not used directly; use init_cov or init_processcov)."""
        return None

    def init_cov(self, iekf):
        """
        Compute initial covariance scaling factors.

        Returns:
            Tensor of shape (output_dim,) with multiplicative scaling factors.
            Values range from 0.1 to 10.0.
        """
        device = self.factor_initial_covariance.weight.device
        alpha = self.factor_initial_covariance(
            torch.ones(1, dtype=torch.float64, device=device)
        ).squeeze()
        beta = 10 ** (self.tanh(alpha))
        return beta

    def init_processcov(self, iekf):
        """
        Compute process noise covariance scaling factors.

        Returns:
            Tensor of shape (output_dim,) with multiplicative scaling factors.
            Values range from 0.1 to 10.0.
        """
        device = self.factor_process_covariance.weight.device
        alpha = self.factor_process_covariance(
            torch.ones(1, dtype=torch.float64, device=device)
        )
        beta = 10 ** (self.tanh(alpha))
        return beta

    def get_output_dim(self):
        return self.output_dim
