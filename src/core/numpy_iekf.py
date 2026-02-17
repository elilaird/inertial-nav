"""
NumPy implementation of Invariant Extended Kalman Filter (IEKF).

This module provides a fast, NumPy-based IEKF for runtime inference on
IMU data using zero lateral/vertical velocity constraints.
"""

import numpy as np
np.set_printoptions(precision=2)

from src.core.base_filter import BaseFilter
from src.utils.geometry import (
    so3_exp, so3_left_jacobian, normalize_rot,
    from_rpy, to_rpy, skew_symmetric
)


class NumPyIEKF(BaseFilter):
    """
    Invariant Extended Kalman Filter implemented in NumPy.

    This implementation is optimized for fast inference and uses zero
    lateral and vertical velocity constraints for measurement updates.

    State vector: [Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i]
    - Rot: 3x3 rotation matrix (world to IMU)
    - v: 3D velocity (world frame)
    - p: 3D position (world frame)
    - b_omega: gyroscope bias (3D)
    - b_acc: accelerometer bias (3D)
    - Rot_c_i: car to IMU rotation (3x3)
    - t_c_i: car to IMU translation (3D)

    Covariance is 21-dimensional on the manifold tangent space.
    """

    # Identity matrices
    Id2 = np.eye(2)
    Id3 = np.eye(3)
    Id6 = np.eye(6)
    IdP = np.eye(21)

    class Parameters:
        """Default filter parameters."""

        g = np.array([0, 0, -9.80665])
        """Gravity vector (m/s^2)"""

        P_dim = 21
        """Covariance dimension"""

        Q_dim = 18
        """Process noise covariance dimension"""

        # Process noise covariances
        cov_omega = 1e-3
        """Gyro covariance"""
        cov_acc = 1e-2
        """Accelerometer covariance"""
        cov_b_omega = 6e-9
        """Gyro bias covariance"""
        cov_b_acc = 2e-4
        """Accelerometer bias covariance"""
        cov_Rot_c_i = 1e-9
        """Car to IMU orientation covariance"""
        cov_t_c_i = 1e-9
        """Car to IMU translation covariance"""

        # Measurement noise covariances
        cov_lat = 0.2
        """Zero lateral velocity covariance"""
        cov_up = 300
        """Zero vertical velocity covariance"""

        # Initial state covariances
        cov_Rot0 = 1e-3
        """Initial pitch and roll covariance"""
        cov_b_omega0 = 6e-3
        """Initial gyro bias covariance"""
        cov_b_acc0 = 4e-3
        """Initial accelerometer bias covariance"""
        cov_v0 = 1e-1
        """Initial velocity covariance"""
        cov_Rot_c_i0 = 1e-6
        """Initial car to IMU pitch and roll covariance"""
        cov_t_c_i0 = 5e-3
        """Initial car to IMU translation covariance"""

        # Numerical parameters
        n_normalize_rot = 100
        """Timesteps before normalizing orientation"""
        n_normalize_rot_c_i = 1000
        """Timesteps before normalizing car to IMU orientation"""

        verbose = False
        """Enable verbose logging"""

        def __init__(self, **kwargs):
            self.set(**kwargs)

        def set(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, parameter_class=None):
        """
        Initialize NumPy IEKF.

        Args:
            parameter_class: Parameter class (default: NumPyIEKF.Parameters)
        """
        super().__init__(parameter_class)

        # Initialize parameters from class or use defaults
        if parameter_class is None:
            self.filter_parameters = NumPyIEKF.Parameters()
        else:
            self.filter_parameters = parameter_class()

        # Copy parameters to instance attributes
        self.set_param_attr()

        # Build process noise covariance matrix
        self._build_Q()

    def _build_Q(self):
        """Build process noise covariance matrix from parameters."""
        self.Q = np.diag([
            self.cov_omega, self.cov_omega, self.cov_omega,
            self.cov_acc, self.cov_acc, self.cov_acc,
            self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
            self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
            self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
            self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i
        ])

    def run(self, t, u, measurements_covs, v_mes, p_mes, N, ang0):
        """
        Run the filter on a sequence of IMU measurements.

        Args:
            t: Timestamps (N,)
            u: IMU measurements (N, 6) - [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
            measurements_covs: Measurement covariances (N, 2) - [lateral, vertical]
            v_mes: Ground truth velocities for initialization (N, 3)
            p_mes: Ground truth positions (N, 3)
            N: Number of timesteps (None = all)
            ang0: Initial orientation [roll, pitch, yaw] (3,)

        Returns:
            Tuple of (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
        """
        dt = t[1:] - t[:-1]  # Time differences (s)
        if N is None:
            N = u.shape[0]

        # Initialize state
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(
            dt, u, p_mes, v_mes, ang0, N
        )

        # Main filter loop
        for i in range(1, N):
            # Propagation step (prediction)
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.propagate(
                    Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1],
                    Rot_c_i[i-1], t_c_i[i-1], P, u[i], dt[i-1]
                )

            # Update step (correction)
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(
                    Rot[i], v[i], p[i], b_omega[i], b_acc[i],
                    Rot_c_i[i], t_c_i[i], P, u[i], i, measurements_covs[i]
                )

            # Correct numerical drift periodically
            if i % self.n_normalize_rot == 0:
                Rot[i] = normalize_rot(Rot[i])
            if i % self.n_normalize_rot_c_i == 0:
                Rot_c_i[i] = normalize_rot(Rot_c_i[i])

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, ang0, N):
        """Initialize filter state and covariance."""
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.init_saved_state(dt, N, ang0)
        Rot[0] = from_rpy(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]
        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        """
        Initialize state covariance matrix.

        Returns:
            Initial covariance P0 (21, 21)
        """
        P = np.zeros((self.P_dim, self.P_dim))
        P[:2, :2] = self.cov_Rot0 * self.Id2  # No yaw error
        P[3:5, 3:5] = self.cov_v0 * self.Id2
        P[9:12, 9:12] = self.cov_b_omega0 * self.Id3
        P[12:15, 12:15] = self.cov_b_acc0 * self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0 * self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0 * self.Id3
        return P

    def init_saved_state(self, dt, N, ang0):
        """Allocate memory for state trajectory."""
        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        Rot_c_i = np.zeros((N, 3, 3))
        t_c_i = np.zeros((N, 3))
        Rot_c_i[0] = np.eye(3)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev,
                  Rot_c_i_prev, t_c_i_prev, P_prev, u, dt):
        """
        Propagate state forward (prediction step).

        Uses IMU measurements to predict next state via dead reckoning.
        """
        # Compute acceleration in world frame
        acc = Rot_prev.dot(u[3:6] - b_acc_prev) + self.g

        # Update velocity and position
        v = v_prev + acc * dt
        p = p_prev + v_prev * dt + 0.5 * acc * dt**2

        # Update orientation
        omega = u[:3] - b_omega_prev
        Rot = Rot_prev.dot(so3_exp(omega * dt))

        # Biases and extrinsics remain constant
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev
        t_c_i = t_c_i_prev

        # Propagate covariance
        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev,
                               b_omega_prev, b_acc_prev, u, dt)

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev,
                      b_omega_prev, b_acc_prev, u, dt):
        """Propagate covariance matrix."""
        F = np.zeros((self.P_dim, self.P_dim))
        G = np.zeros((self.P_dim, self.Q_dim))

        v_skew_rot = skew_symmetric(v_prev).dot(Rot_prev)
        p_skew_rot = skew_symmetric(p_prev).dot(Rot_prev)

        # Build state transition matrix F
        F[3:6, :3] = skew_symmetric(self.g)
        F[6:9, 3:6] = self.Id3
        F[3:6, 12:15] = -Rot_prev
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot

        # Build noise matrix G
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        G[3:6, 3:6] = Rot_prev
        G[9:15, 6:12] = self.Id6
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        # Scale by timestep
        F = F * dt
        G = G * dt

        # Compute state transition matrix using Taylor expansion
        F_square = F.dot(F)
        F_cube = F_square.dot(F)
        Phi = self.IdP + F + 0.5 * F_square + (1.0/6.0) * F_cube

        # Propagate covariance
        P = Phi.dot(P_prev + G.dot(self.Q).dot(G.T)).dot(Phi.T)

        return P

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i,
               measurement_cov):
        """
        Update state using zero velocity constraints (correction step).

        Applies lateral and vertical zero velocity constraints.
        """
        # Orientation of body frame
        Rot_body = Rot.dot(Rot_c_i)

        # Velocity in IMU frame
        v_imu = Rot.T.dot(v)

        # Velocity in body frame
        v_body = Rot_c_i.T.dot(v_imu)

        # Add velocity contribution from IMU offset
        v_body += skew_symmetric(t_c_i).dot(u[:3] - b_omega)

        Omega = skew_symmetric(u[:3] - b_omega)

        # Measurement Jacobian
        H_v_imu = Rot_c_i.T.dot(skew_symmetric(v_imu))
        H_t_c_i = -skew_symmetric(t_c_i)

        H = np.zeros((2, self.P_dim))
        H[:, 3:6] = Rot_body.T[1:]  # Lateral and vertical components
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]

        # Innovation (measurement residual)
        r = -v_body[1:]  # Expect zero lateral and vertical velocity

        # Measurement noise covariance
        R = np.diag(measurement_cov)

        # Perform state and covariance update
        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)

        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        """
        Perform Kalman update on state and covariance.

        Uses invariant error formulation on SE_2(3) manifold.
        """
        # Kalman gain
        S = H.dot(P).dot(H.T) + R
        K = (np.linalg.solve(S, P.dot(H.T).T)).T

        # Compute state correction
        dx = K.dot(r)

        # Update SE_2(3) state (rotation, velocity, position)
        dR, dxi = NumPyIEKF.se23_exp(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.dot(Rot)
        v_up = dR.dot(v) + dv
        p_up = dR.dot(p) + dp

        # Update biases
        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        # Update extrinsics
        dR = so3_exp(dx[15:18])
        Rot_c_i_up = dR.dot(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        # Update covariance (Joseph form for numerical stability)
        I_KH = NumPyIEKF.IdP - K.dot(H)
        P_up = I_KH.dot(P).dot(I_KH.T) + K.dot(R).dot(K.T)
        P_up = (P_up + P_up.T) / 2  # Ensure symmetry

        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def se23_exp(xi):
        """
        SE_2(3) exponential map.

        Converts element of se_2(3) Lie algebra to SE_2(3) group element.

        Args:
            xi: 9D vector [phi, v, p] where phi is rotation, v is velocity, p is position

        Returns:
            Tuple of (Rot, x) where:
                Rot: 3x3 rotation matrix
                x: 3x2 matrix of [velocity, position] vectors
        """
        phi = xi[:3]
        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.abs(angle) < 1e-8:
            skew_phi = skew_symmetric(phi)
            J = NumPyIEKF.Id3 + 0.5 * skew_phi
            Rot = NumPyIEKF.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = skew_symmetric(axis)
            s = np.sin(angle)
            c = np.cos(angle)

            # Left Jacobian
            J = (s / angle) * NumPyIEKF.Id3 \
                + (1 - s / angle) * np.outer(axis, axis) \
                + ((1 - c) / angle) * skew_axis

            # Rotation
            Rot = c * NumPyIEKF.Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis

        # Apply Jacobian to velocity and position parts
        x = J.dot(xi[3:].reshape(-1, 3).T)

        return Rot, x

    @staticmethod
    def rot_from_2_vectors(v1, v2):
        """
        Compute rotation matrix that aligns v1 with v2.

        Args:
            v1: First vector (3,)
            v2: Second vector (3,)

        Returns:
            3x3 rotation matrix
        """
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        v = np.cross(v1, v2)
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(v)

        if sinang < 1e-10:
            # Vectors are parallel
            return NumPyIEKF.Id3

        Rot = NumPyIEKF.Id3 + skew_symmetric(v) + \
              skew_symmetric(v).dot(skew_symmetric(v)) * (1 - cosang) / (sinang**2)
        Rot = normalize_rot(Rot)

        return Rot

    def set_learned_covariance(self, torch_iekf):
        """
        Load learned covariances from PyTorch IEKF.

        This method updates the filter's process and initial covariances
        using the values learned by the neural networks.

        Args:
            torch_iekf: PyTorch IEKF instance with trained networks
        """
        torch_iekf.set_Q()
        self.Q = torch_iekf.Q.cpu().detach().numpy()

        beta = torch_iekf.initprocesscov_net.init_cov(torch_iekf)\
            .detach().cpu().numpy()

        self.cov_Rot0 *= beta[0]
        self.cov_v0 *= beta[1]
        self.cov_b_omega0 *= beta[2]
        self.cov_b_acc0 *= beta[3]
        self.cov_Rot_c_i0 *= beta[4]
        self.cov_t_c_i0 *= beta[5]


# Alias for backward compatibility
NUMPYIEKF = NumPyIEKF
