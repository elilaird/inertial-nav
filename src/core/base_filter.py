"""
Base filter abstraction for IEKF implementations.

This module defines the abstract interface that all IEKF implementations
(NumPy, PyTorch) must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for Invariant Extended Kalman Filter implementations.

    This class defines the interface that all filter implementations must follow,
    ensuring consistent API across NumPy and PyTorch versions.
    """

    def __init__(self, parameter_class=None):
        """
        Initialize the filter.

        Args:
            parameter_class: Class containing filter parameters (optional)
        """
        self.filter_parameters = None

    @abstractmethod
    def run(self, t: np.ndarray, u: np.ndarray, measurements_covs: np.ndarray,
            v_mes: np.ndarray, p_mes: np.ndarray, N: Optional[int],
            ang0: np.ndarray) -> Tuple:
        """
        Run the filter on a sequence of IMU measurements.

        Args:
            t: Timestamps (N,)
            u: IMU measurements (N, 6) - [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
            measurements_covs: Measurement covariances (N, 2) - [lateral, vertical]
            v_mes: Ground truth velocities for initialization (N, 3)
            p_mes: Ground truth positions for reference (N, 3)
            N: Number of timesteps (None = all)
            ang0: Initial orientation [roll, pitch, yaw] (3,)

        Returns:
            Tuple of (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i):
                Rot: Rotation matrices (N, 3, 3)
                v: Velocities (N, 3)
                p: Positions (N, 3)
                b_omega: Gyroscope biases (N, 3)
                b_acc: Accelerometer biases (N, 3)
                Rot_c_i: Car-to-IMU rotation (N, 3, 3)
                t_c_i: Car-to-IMU translation (N, 3)
        """
        pass

    @abstractmethod
    def propagate(self, Rot: np.ndarray, v: np.ndarray, p: np.ndarray,
                  b_omega: np.ndarray, b_acc: np.ndarray,
                  Rot_c_i: np.ndarray, t_c_i: np.ndarray,
                  P: np.ndarray, u: np.ndarray, dt: float) -> Tuple:
        """
        Propagate state forward using IMU measurements (prediction step).

        Args:
            Rot: Current rotation matrix (3, 3)
            v: Current velocity (3,)
            p: Current position (3,)
            b_omega: Current gyroscope bias (3,)
            b_acc: Current accelerometer bias (3,)
            Rot_c_i: Current car-to-IMU rotation (3, 3)
            t_c_i: Current car-to-IMU translation (3,)
            P: Current covariance matrix (21, 21)
            u: IMU measurement (6,) - [gyro, acc]
            dt: Time step (seconds)

        Returns:
            Tuple of propagated state (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P)
        """
        pass

    @abstractmethod
    def update(self, Rot: np.ndarray, v: np.ndarray, p: np.ndarray,
               b_omega: np.ndarray, b_acc: np.ndarray,
               Rot_c_i: np.ndarray, t_c_i: np.ndarray,
               P: np.ndarray, u: np.ndarray, i: int,
               measurement_cov: np.ndarray) -> Tuple:
        """
        Update state using zero velocity constraints (correction step).

        Args:
            Rot: Propagated rotation matrix (3, 3)
            v: Propagated velocity (3,)
            p: Propagated position (3,)
            b_omega: Propagated gyroscope bias (3,)
            b_acc: Propagated accelerometer bias (3,)
            Rot_c_i: Propagated car-to-IMU rotation (3, 3)
            t_c_i: Propagated car-to-IMU translation (3,)
            P: Propagated covariance matrix (21, 21)
            u: IMU measurement (6,)
            i: Current timestep index
            measurement_cov: Measurement noise covariances (2,) - [lateral, vertical]

        Returns:
            Tuple of updated state (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P)
        """
        pass

    @abstractmethod
    def init_covariance(self) -> np.ndarray:
        """
        Initialize the state covariance matrix.

        Returns:
            Initial covariance matrix P0 (21, 21)
        """
        pass

    def set_param_attr(self):
        """
        Set filter attributes from parameter class.

        This method copies all non-callable attributes from filter_parameters
        to the filter instance.
        """
        if self.filter_parameters is None:
            return

        # Get list of non-callable attributes
        attr_list = [a for a in dir(self.filter_parameters)
                     if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]

        # Copy attributes to filter instance
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get filter state as a dictionary (for checkpointing).

        Returns:
            Dictionary containing filter state and parameters
        """
        return {
            'filter_parameters': self.filter_parameters,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load filter state from a dictionary.

        Args:
            state_dict: Dictionary containing filter state and parameters
        """
        self.filter_parameters = state_dict.get('filter_parameters', None)
        if self.filter_parameters is not None:
            self.set_param_attr()
