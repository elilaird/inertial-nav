"""
Abstract base class for covariance prediction networks.

All covariance networks (CNN, LSTM, Transformer) implement this interface
so they can be swapped into the IEKF filter without code changes.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseCovarianceNet(nn.Module, ABC):
    """
    Abstract base class for networks that predict covariance matrices.

    Subclasses must implement forward() and get_output_dim().
    This enables swapping different architectures (CNN, LSTM, Transformer)
    via configuration without modifying the filter code.
    """

    @abstractmethod
    def forward(self, u, iekf):
        """
        Forward pass: predict covariance scaling factors from IMU data.

        Args:
            u: IMU input tensor. Shape depends on the specific network.
            iekf: Reference to the IEKF filter instance (provides baseline covariances).

        Returns:
            Predicted covariance values (shape depends on network type).
        """
        pass

    @abstractmethod
    def get_output_dim(self):
        """
        Return the output dimension of this network.

        Returns:
            int: Number of output covariance parameters.
        """
        pass
