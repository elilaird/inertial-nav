"""
PyTorch implementation of Invariant Extended Kalman Filter (IEKF).

This module provides a PyTorch-based IEKF for training with neural networks
that predict adaptive covariances. The filter can operate with or without
learned covariance networks.
"""

import torch
import numpy as np
import os
from termcolor import cprint

from src.core.numpy_iekf import NumPyIEKF
from src.models import get_model


def isclose(mat1, mat2, tol=1e-10):
    """Check if two tensors are close within tolerance."""
    return (mat1 - mat2).abs().lt(tol)


class TorchIEKF(torch.nn.Module, NumPyIEKF):
    """
    PyTorch implementation of IEKF with optional learned covariances.

    This implementation allows gradient flow for training neural networks
    that predict adaptive measurement and process noise covariances.

    The filter maintains the same mathematical structure as NumPyIEKF but
    uses PyTorch tensors for automatic differentiation.
    """

    # PyTorch identity matrices
    Id1 = torch.eye(1).double()
    Id2 = torch.eye(2).double()
    Id3 = torch.eye(3).double()
    Id6 = torch.eye(6).double()
    IdP = torch.eye(21).double()

    def __init__(self, parameter_class=None):
        """
        Initialize PyTorch IEKF.

        Args:
            parameter_class: Parameter class (default: NumPyIEKF.Parameters)
        """
        torch.nn.Module.__init__(self)

        # Initialize parameters (don't call NumPyIEKF.__init__ to avoid numpy Q)
        if parameter_class is None:
            self.filter_parameters = NumPyIEKF.Parameters()
        else:
            self.filter_parameters = parameter_class()

        # Copy parameters and build PyTorch matrices
        self.set_param_attr()

        # Input normalization parameters (set during training)
        self.u_loc = None
        self.u_std = None

        # Note: cov0_measurement is set by set_param_attr() above

        # Network components (None = use defaults)
        self.initprocesscov_net = None
        self.mes_net = None
        self.dynamics_net = None

        # Override IdP with PyTorch version
        self.IdP = torch.eye(self.P_dim).double()

    @classmethod
    def build_from_cfg(cls, cfg, parameter_class=None):
        """
        Build TorchIEKF with networks from a Hydra config.

        Args:
            cfg: OmegaConf config with 'networks' key containing network specs.
            parameter_class: Optional parameter class override.

        Returns:
            Configured TorchIEKF instance with networks attached.
        """
        iekf = cls(parameter_class=parameter_class)

        networks_cfg = cfg.get("networks", {})

        # Default type names when config omits an explicit 'type' key
        _DEFAULT_TYPES = {
            "init_process_cov": "InitProcessCovNet",
            "measurement_cov": "MeasurementCovNet",
            "dynamics": "NeuralODEDynamics",
        }

        # Build init/process covariance network
        ipc_cfg = networks_cfg.get("init_process_cov", {})
        if ipc_cfg.get("enabled", False):
            type_name = ipc_cfg.get("type", _DEFAULT_TYPES["init_process_cov"])
            iekf.initprocesscov_net = cls._build_network(
                type_name, ipc_cfg.get("architecture", {})
            )

        # Build measurement covariance network
        mes_cfg = networks_cfg.get("measurement_cov", {})
        if mes_cfg.get("enabled", False):
            type_name = mes_cfg.get("type", _DEFAULT_TYPES["measurement_cov"])
            iekf.mes_net = cls._build_network(
                type_name, mes_cfg.get("architecture", {})
            )

        # Build optional learned dynamics network
        dyn_cfg = networks_cfg.get("dynamics", {})
        if dyn_cfg.get("enabled", False):
            type_name = dyn_cfg.get("type", _DEFAULT_TYPES["dynamics"])
            iekf.dynamics_net = cls._build_network(
                type_name, dyn_cfg.get("architecture", {})
            )

        return iekf

    @staticmethod
    def _build_network(type_name, architecture_cfg):
        """
        Instantiate a network from the model registry.

        Args:
            type_name: Registered model class name (e.g., "MeasurementCovNet")
            architecture_cfg: Dict/DictConfig of constructor kwargs.

        Returns:
            Instantiated network module.
        """
        import inspect

        model_cls = get_model(type_name)
        kwargs = dict(architecture_cfg) if architecture_cfg else {}

        # Filter kwargs to only those accepted by the constructor,
        # so extra config keys (e.g. 'hidden_features') don't cause errors.
        sig = inspect.signature(model_cls.__init__)
        params = sig.parameters
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            valid_keys = set(params.keys()) - {"self"}
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return model_cls(**kwargs)

    def set_param_attr(self):
        """Set filter attributes from parameter class."""
        # Get list of non-callable attributes
        if self.filter_parameters is None:
            return

        attr_list = [
            a
            for a in dir(self.filter_parameters)
            if not a.startswith("__")
            and not callable(getattr(self.filter_parameters, a))
        ]

        # Copy attributes to filter instance
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        # Convert gravity to torch if needed
        if hasattr(self, "g") and isinstance(self.g, np.ndarray):
            self.g = torch.from_numpy(self.g).double()

        # Build PyTorch-specific matrices
        self._build_torch_Q()
        if hasattr(self, "cov_lat") and hasattr(self, "cov_up"):
            self.cov0_measurement = torch.Tensor(
                [self.cov_lat, self.cov_up]
            ).double()

    def _build_torch_Q(self):
        """Build process noise covariance matrix (PyTorch version)."""
        self.Q = torch.diag(
            torch.Tensor(
                [
                    self.cov_omega,
                    self.cov_omega,
                    self.cov_omega,
                    self.cov_acc,
                    self.cov_acc,
                    self.cov_acc,
                    self.cov_b_omega,
                    self.cov_b_omega,
                    self.cov_b_omega,
                    self.cov_b_acc,
                    self.cov_b_acc,
                    self.cov_b_acc,
                    self.cov_Rot_c_i,
                    self.cov_Rot_c_i,
                    self.cov_Rot_c_i,
                    self.cov_t_c_i,
                    self.cov_t_c_i,
                    self.cov_t_c_i,
                ]
            )
        ).double()

    def run(self, t, u, measurements_covs, v_mes, p_mes, N, ang0):
        """
        Run the filter on a sequence of IMU measurements.

        Args:
            t: Timestamps (N,) - PyTorch tensor
            u: IMU measurements (N, 6) - PyTorch tensor
            measurements_covs: Measurement covariances (N, 2) - PyTorch tensor
            v_mes: Ground truth velocities (N, 3) - PyTorch tensor
            p_mes: Ground truth positions (N, 3) - PyTorch tensor
            N: Number of timesteps (None = all)
            ang0: Initial orientation [roll, pitch, yaw] (3,)

        Returns:
            Tuple of (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
        """
        dt = t[1:] - t[:-1]
        if N is None:
            N = u.shape[0]

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(
            dt, u, p_mes, v_mes, N, ang0
        )

        for i in range(1, N):
            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = (
                self.propagate(
                    Rot[i - 1],
                    v[i - 1],
                    p[i - 1],
                    b_omega[i - 1],
                    b_acc[i - 1],
                    Rot_c_i[i - 1],
                    t_c_i[i - 1],
                    P,
                    u[i],
                    dt[i - 1],
                )
            )

            (
                Rot[i],
                v[i],
                p[i],
                b_omega[i],
                b_acc[i],
                Rot_c_i[i],
                t_c_i[i],
                P,
            ) = self.update(
                Rot_i,
                v_i,
                p_i,
                b_omega_i,
                b_acc_i,
                Rot_c_i_i,
                t_c_i_i,
                P_i,
                u[i],
                i,
                measurements_covs[i],
            )

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
        """Initialize filter state and covariance (PyTorch version)."""
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.init_saved_state(
            dt, N, ang0
        )
        Rot[0] = self.from_rpy_torch(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]
        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        """
        Initialize state covariance matrix (PyTorch version).

        If initprocesscov_net is available, uses learned scaling factors.
        Otherwise uses default covariances.
        """
        P = torch.zeros(self.P_dim, self.P_dim).double()

        # Get scaling factors from network if available
        if self.initprocesscov_net is not None:
            beta = self.initprocesscov_net.init_cov(self)
        else:
            beta = torch.ones(6).double()

        P[:2, :2] = self.cov_Rot0 * beta[0] * self.Id2
        P[3:5, 3:5] = self.cov_v0 * beta[1] * self.Id2
        P[9:12, 9:12] = self.cov_b_omega0 * beta[2] * self.Id3
        P[12:15, 12:15] = self.cov_b_acc0 * beta[3] * self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0 * beta[4] * self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0 * beta[5] * self.Id3

        return P

    def init_saved_state(self, dt, N, ang0):
        """Allocate memory for state trajectory (PyTorch version)."""
        Rot = dt.new_zeros(N, 3, 3)
        v = dt.new_zeros(N, 3)
        p = dt.new_zeros(N, 3)
        b_omega = dt.new_zeros(N, 3)
        b_acc = dt.new_zeros(N, 3)
        Rot_c_i = dt.new_zeros(N, 3, 3)
        t_c_i = dt.new_zeros(N, 3)
        Rot_c_i[0] = torch.eye(3).double()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(
        self,
        Rot_prev,
        v_prev,
        p_prev,
        b_omega_prev,
        b_acc_prev,
        Rot_c_i_prev,
        t_c_i_prev,
        P_prev,
        u,
        dt,
    ):
        """
        Propagate state forward (PyTorch version).

        If dynamics_net is set, uses learned dynamics for velocity/position.
        Otherwise uses classical inertial kinematics.
        """
        Rot_prev = Rot_prev.clone()

        if self.dynamics_net is not None:
            # Learned dynamics for velocity and position
            v, p = self.dynamics_net(
                v_prev, p_prev, Rot_prev, u, b_acc_prev, self.g, dt
            )
        else:
            # Classical inertial kinematics
            acc_b = u[3:6] - b_acc_prev
            acc = Rot_prev.mv(acc_b) + self.g
            v = v_prev + acc * dt
            p = p_prev + v_prev.clone() * dt + 0.5 * acc * dt**2

        # Rotation always uses classical SO(3) integration
        omega = (u[:3] - b_omega_prev) * dt
        Rot = Rot_prev.mm(self.so3exp_torch(omega))

        # Biases and extrinsics remain constant
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        Rot_c_i = Rot_c_i_prev.clone()
        t_c_i = t_c_i_prev

        # Propagate covariance (always classical for now)
        P = self.propagate_cov_torch(
            P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt
        )

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov_torch(
        self, P, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt
    ):
        """Propagate covariance matrix (PyTorch version)."""
        F = P.new_zeros(self.P_dim, self.P_dim)
        G = P.new_zeros(self.P_dim, self.Q.shape[0])
        Q = self.Q.clone()

        v_skew_rot = self.skew_torch(v_prev).mm(Rot_prev)
        p_skew_rot = self.skew_torch(p_prev).mm(Rot_prev)

        # Build state transition matrix F
        F[3:6, :3] = self.skew_torch(self.g)
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
        G[9:12, 6:9] = self.Id3
        G[12:15, 9:12] = self.Id3
        G[15:18, 12:15] = self.Id3
        G[18:21, 15:18] = self.Id3

        # Scale by timestep
        F = F * dt
        G = G * dt

        # Compute state transition matrix
        F_square = F.mm(F)
        F_cube = F_square.mm(F)
        Phi = self.IdP + F + 0.5 * F_square + (1.0 / 6.0) * F_cube

        # Propagate covariance
        P_new = Phi.mm(P + G.mm(Q).mm(G.t())).mm(Phi.t())

        return P_new

    def update(
        self,
        Rot,
        v,
        p,
        b_omega,
        b_acc,
        Rot_c_i,
        t_c_i,
        P,
        u,
        i,
        measurement_cov,
    ):
        """Update state using zero velocity constraints (PyTorch version)."""
        # Orientation of body frame
        Rot_body = Rot.mm(Rot_c_i)

        # Velocity in IMU frame
        v_imu = Rot.t().mv(v)

        # Velocity in body frame
        omega = u[:3] - b_omega
        v_body = Rot_c_i.t().mv(v_imu) + self.skew_torch(t_c_i).mv(omega)
        Omega = self.skew_torch(omega)

        # Measurement Jacobian
        H_v_imu = Rot_c_i.t().mm(self.skew_torch(v_imu))
        H_t_c_i = self.skew_torch(t_c_i)

        H = P.new_zeros(2, self.P_dim)
        H[:, 3:6] = Rot_body.t()[1:]
        H[:, 15:18] = H_v_imu[1:]
        H[:, 9:12] = H_t_c_i[1:]
        H[:, 18:21] = -Omega[1:]

        # Innovation
        r = -v_body[1:]

        # Measurement noise covariance
        R = torch.diag(measurement_cov)

        # Perform update
        (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        ) = self.state_and_cov_update_torch(
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
        )

        return (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        )

    @staticmethod
    def state_and_cov_update_torch(
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R
    ):
        """Perform Kalman update (PyTorch version)."""
        # Kalman gain
        S = H.mm(P).mm(H.t()) + R
        try:
            # Modern PyTorch
            K = torch.linalg.solve(S, P.mm(H.t()).t()).t()
        except AttributeError:
            # Older PyTorch
            Kt, _ = torch.gesv(P.mm(H.t()).t(), S)
            K = Kt.t()

        # Compute state correction
        dx = K.mv(r.view(-1))

        # Update SE_2(3) state
        dR, dxi = TorchIEKF.se23_exp_torch(dx[:9])
        dv = dxi[:, 0]
        dp = dxi[:, 1]
        Rot_up = dR.mm(Rot)
        v_up = dR.mv(v) + dv
        p_up = dR.mv(p) + dp

        # Update biases
        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        # Update extrinsics
        dR = TorchIEKF.so3exp_torch(dx[15:18])
        Rot_c_i_up = dR.mm(Rot_c_i)
        t_c_i_up = t_c_i + dx[18:21]

        # Update covariance (Joseph form)
        I_KH = TorchIEKF.IdP - K.mm(H)
        P_upprev = I_KH.mm(P).mm(I_KH.t()) + K.mm(R).mm(K.t())
        P_up = (P_upprev + P_upprev.t()) / 2

        return (
            Rot_up,
            v_up,
            p_up,
            b_omega_up,
            b_acc_up,
            Rot_c_i_up,
            t_c_i_up,
            P_up,
        )

    # ========== PyTorch Geometry Utilities ==========

    @staticmethod
    def skew_torch(x):
        """Skew-symmetric matrix (PyTorch version)."""
        zero = x.new_zeros(1).squeeze()
        return torch.stack(
            [
                torch.stack([zero, -x[2], x[1]]),
                torch.stack([x[2], zero, -x[0]]),
                torch.stack([-x[1], x[0], zero]),
            ]
        )

    @staticmethod
    def so3exp_torch(phi):
        """SO(3) exponential map (PyTorch version)."""
        angle = phi.norm()

        if isclose(angle, torch.tensor(0.0)):
            skew_phi = TorchIEKF.skew_torch(phi)
            return TorchIEKF.Id3 + skew_phi

        axis = phi / angle
        skew_axis = TorchIEKF.skew_torch(axis)
        c = angle.cos()
        s = angle.sin()

        return (
            c * TorchIEKF.Id3
            + (1 - c) * TorchIEKF.outer(axis, axis)
            + s * skew_axis
        )

    @staticmethod
    def se23_exp_torch(xi):
        """SE_2(3) exponential map (PyTorch version)."""
        phi = xi[:3]
        angle = torch.norm(phi)

        if isclose(angle, torch.tensor(0.0)):
            skew_phi = TorchIEKF.skew_torch(phi)
            J = TorchIEKF.Id3 + 0.5 * skew_phi
            Rot = TorchIEKF.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = TorchIEKF.skew_torch(axis)
            s = torch.sin(angle)
            c = torch.cos(angle)

            J = (
                (s / angle) * TorchIEKF.Id3
                + (1 - s / angle) * TorchIEKF.outer(axis, axis)
                + ((1 - c) / angle) * skew_axis
            )
            Rot = (
                c * TorchIEKF.Id3
                + (1 - c) * TorchIEKF.outer(axis, axis)
                + s * skew_axis
            )

        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    @staticmethod
    def from_rpy_torch(roll, pitch, yaw):
        """Convert RPY to rotation matrix (PyTorch version)."""
        return TorchIEKF.rotz_torch(yaw).mm(
            TorchIEKF.roty_torch(pitch).mm(TorchIEKF.rotx_torch(roll))
        )

    @staticmethod
    def rotx_torch(t):
        """Rotation around x-axis (PyTorch version)."""
        c = torch.cos(t)
        s = torch.sin(t)
        return torch.Tensor([[1, 0, 0], [0, c, -s], [0, s, c]]).double()

    @staticmethod
    def roty_torch(t):
        """Rotation around y-axis (PyTorch version)."""
        c = torch.cos(t)
        s = torch.sin(t)
        return torch.Tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]).double()

    @staticmethod
    def rotz_torch(t):
        """Rotation around z-axis (PyTorch version)."""
        c = torch.cos(t)
        s = torch.sin(t)
        return torch.Tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).double()

    @staticmethod
    def outer(a, b):
        """Outer product (PyTorch version)."""
        return torch.ger(a, b)

    @staticmethod
    def normalize_rot_torch(Rot):
        """Normalize rotation matrix using SVD (PyTorch version)."""
        U, _, V = torch.svd(Rot)
        S = torch.eye(3).double()
        S[2, 2] = torch.det(U) * torch.det(V)
        return U.mm(S).mm(V.t())

    # ========== Network Integration Methods (Phase 2) ==========

    def forward_nets(self, u):
        """
        Forward pass through covariance prediction networks.

        Args:
            u: IMU measurements (N, 6)

        Returns:
            Predicted measurement covariances (N, 2)
        """
        if self.mes_net is None:
            # Use default covariances if no network
            return self.cov0_measurement.unsqueeze(0).repeat(u.shape[0], 1)

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        measurements_covs = self.mes_net(u_n, self)
        return measurements_covs

    def set_Q(self):
        """
        Update process noise covariance using learned scaling factors.

        If initprocesscov_net is available, uses its predictions.
        Otherwise uses default covariances.
        """
        # Build base Q
        self._build_torch_Q()

        if self.initprocesscov_net is None:
            return

        # Apply learned scaling
        beta = self.initprocesscov_net.init_processcov(self)
        self.Q = torch.zeros(self.Q.shape[0], self.Q.shape[0]).double()
        self.Q[:3, :3] = self.cov_omega * beta[0] * self.Id3
        self.Q[3:6, 3:6] = self.cov_acc * beta[1] * self.Id3
        self.Q[6:9, 6:9] = self.cov_b_omega * beta[2] * self.Id3
        self.Q[9:12, 9:12] = self.cov_b_acc * beta[3] * self.Id3
        self.Q[12:15, 12:15] = self.cov_Rot_c_i * beta[4] * self.Id3
        self.Q[15:18, 15:18] = self.cov_t_c_i * beta[5] * self.Id3

    def normalize_u(self, u):
        """Normalize IMU inputs using dataset statistics."""
        if self.u_loc is None or self.u_std is None:
            return u
        return (u - self.u_loc) / self.u_std

    def get_normalize_u(self, dataset):
        """Load normalization parameters from dataset."""
        self.u_loc = dataset.normalize_factors["u_loc"].double()
        self.u_std = dataset.normalize_factors["u_std"].double()

    def load(self, args, dataset):
        """
        Load trained network weights.

        Args:
            args: Arguments containing path_temp
            dataset: Dataset containing normalization factors
        """
        path_iekf = os.path.join(args.path_temp, "iekfnets.p")
        if os.path.isfile(path_iekf):
            mondict = torch.load(path_iekf)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", "green")
        else:
            cprint("IEKF nets NOT loaded", "yellow")
        self.get_normalize_u(dataset)


# Alias for backward compatibility
TORCHIEKF = TorchIEKF
