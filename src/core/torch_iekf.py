"""
PyTorch implementation of Invariant Extended Kalman Filter (IEKF).

This is the single IEKF implementation used for both training (with autograd)
and inference (with torch.no_grad).  All computation is done in PyTorch which
can target CPU or CUDA.
"""

import torch
import os
from termcolor import cprint

from src.models import get_model


def isclose(mat1, mat2, tol=1e-10):
    """Check if two tensors are close within tolerance."""
    return (mat1 - mat2).abs().lt(tol)


class TorchIEKF(torch.nn.Module):
    """
    PyTorch IEKF with optional learned covariance networks.

    Supports automatic differentiation for training neural networks that
    predict adaptive measurement and process noise covariances.

    State vector: [Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i]
    Covariance is 21-dimensional on the manifold tangent space.
    """

    # ------------------------------------------------------------------
    # Default filter parameters
    # ------------------------------------------------------------------
    class Parameters:
        """Default IEKF filter parameters."""

        g = [0, 0, -9.80665]
        """Gravity vector (m/s^2)."""

        P_dim = 21
        """Covariance dimension."""

        Q_dim = 18
        """Process noise covariance dimension."""

        # Process noise covariances
        cov_omega = 1e-3
        cov_acc = 1e-2
        cov_b_omega = 6e-9
        cov_b_acc = 2e-4
        cov_Rot_c_i = 1e-9
        cov_t_c_i = 1e-9

        # Measurement noise covariances
        cov_lat = 0.2
        cov_up = 300

        # Initial state covariances
        cov_Rot0 = 1e-3
        cov_b_omega0 = 6e-3
        cov_b_acc0 = 4e-3
        cov_v0 = 1e-1
        cov_Rot_c_i0 = 1e-6
        cov_t_c_i0 = 5e-3

        # Numerical parameters
        n_normalize_rot = 100
        n_normalize_rot_c_i = 1000
        verbose = False

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parameter_class=None):
        """
        Initialize PyTorch IEKF.

        Args:
            parameter_class: Parameter class (default: TorchIEKF.Parameters)
        """
        super().__init__()

        if parameter_class is None:
            params = self.Parameters()
        else:
            params = parameter_class()

        # Copy scalar parameters as plain attributes
        self._copy_scalar_params(params)

        # Register tensor-valued parameters as buffers
        self.register_buffer("g", torch.tensor(params.g, dtype=torch.float32))
        self.register_buffer(
            "cov0_measurement",
            torch.tensor([params.cov_lat, params.cov_up], dtype=torch.float32),
        )

        # Identity matrix buffers (moved automatically by model.to(device))
        self.register_buffer("Id2", torch.eye(2, dtype=torch.float32))
        self.register_buffer("Id3", torch.eye(3, dtype=torch.float32))
        self.register_buffer("Id6", torch.eye(6, dtype=torch.float32))
        self.register_buffer("IdP", torch.eye(self.P_dim, dtype=torch.float32))

        # Process noise covariance buffer
        self._register_Q()

        # Input normalization parameters (set via get_normalize_u)
        self.register_buffer("u_loc", None)
        self.register_buffer("u_std", None)

        # Optional neural network components
        self.initprocesscov_net = None
        self.mes_net = None
        self.bias_correction_net = None

    def _copy_scalar_params(self, params):
        """Copy non-tensor, non-callable attributes from *params* to self."""
        skip = {"g"}  # handled as buffers
        for name in dir(params):
            if name.startswith("_") or name in skip:
                continue
            val = getattr(params, name)
            if callable(val):
                continue
            setattr(self, name, val)

    def _register_Q(self):
        """Build and register the process noise covariance matrix."""
        q_vals = torch.tensor(
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
            ],
            dtype=torch.float32,
        )
        self.register_buffer("Q", torch.diag(q_vals))

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the model lives on (inferred from registered buffers)."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

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
            "bias_correction": "LearnedBiasCorrectionNet",
        }

        for key, attr in [
            ("init_process_cov", "initprocesscov_net"),
            ("measurement_cov", "mes_net"),
            ("bias_correction", "bias_correction_net"),
        ]:
            net_cfg = networks_cfg.get(key, {})
            if net_cfg.get("enabled", False):
                type_name = net_cfg.get("type", _DEFAULT_TYPES[key])
                net = cls._build_network(
                    type_name, net_cfg.get("architecture", {})
                )
                setattr(iekf, attr, net)

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

    # ------------------------------------------------------------------
    # Filter loop
    # ------------------------------------------------------------------

    def run(
        self,
        t,
        u,
        measurements_covs,
        v_mes,
        p_mes,
        N,
        ang0,
        bias_corrections=None,
    ):
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
            bias_corrections: Optional per-timestep accelerometer bias
                corrections (N, 3) from LearnedBiasCorrectionNet.

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
            bc_i = (
                bias_corrections[i] if bias_corrections is not None else None
            )
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
                    bias_correction=bc_i,
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

    def init_state(self, t, u, v_mes, ang0):
        """
        Build the initial filter state for BPTT chunk-wise execution.

        Unlike ``init_run``, this returns a plain state tuple rather than
        pre-allocated trajectory arrays, so it can be threaded between chunks.

        Args:
            t:      Timestamps for the full sequence (used to get dtype/device).
            u:      IMU measurements (for device reference).
            v_mes:  Ground-truth velocities (N, 3); v_mes[0] used for init.
            ang0:   Initial Euler angles [roll, pitch, yaw].

        Returns:
            State dict with keys: Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P.
        """
        P = self.init_covariance()
        Rot0 = self.from_rpy_torch(ang0[0], ang0[1], ang0[2])
        v0 = v_mes[0].clone()
        p0 = torch.zeros(3, dtype=t.dtype, device=t.device)
        b_omega0 = torch.zeros(3, dtype=t.dtype, device=t.device)
        b_acc0 = torch.zeros(3, dtype=t.dtype, device=t.device)
        Rot_c_i0 = torch.eye(3).float()
        t_c_i0 = t.new_zeros(3).float()
        return dict(
            Rot=Rot0,
            v=v0,
            p=p0,
            b_omega=b_omega0,
            b_acc=b_acc0,
            Rot_c_i=Rot_c_i0,
            t_c_i=t_c_i0,
            P=P,
        )

    @staticmethod
    def detach_state(state):
        """Detach all state tensors from the autograd graph (for BPTT)."""
        return {k: v.detach() for k, v in state.items()}

    def run_chunk(
        self,
        state,
        t_chunk,
        u_chunk,
        measurements_covs_chunk,
        bias_corrections_chunk=None,
    ):
        """
        Run the filter for one chunk of timesteps, starting from ``state``.

        Used for Truncated BPTT: caller detaches ``state`` between chunks
        to limit gradient flow to ``chunk_size`` timesteps.

        Args:
            state:               State dict from ``init_state`` or previous chunk.
            t_chunk:             Timestamps for this chunk (K,).
            u_chunk:             IMU measurements for this chunk (K, 6).
            measurements_covs_chunk: Measurement covariances for this chunk (K, 2).
            bias_corrections_chunk: Optional per-timestep accelerometer bias
                corrections (K, 3) from LearnedBiasCorrectionNet.

        Returns:
            traj:       Tuple (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i),
                        each of shape (K, ...).  Positions are *absolute*
                        (continuing from whatever ``state`` held).
            new_state:  State dict at the end of the chunk (still on graph).
        """
        K = t_chunk.shape[0]
        dt_chunk = t_chunk[1:] - t_chunk[:-1]  # (K-1,)

        # Allocate chunk trajectory tensors
        Rot = torch.zeros(K, 3, 3, dtype=t_chunk.dtype, device=t_chunk.device)
        v = torch.zeros(K, 3, dtype=t_chunk.dtype, device=t_chunk.device)
        p = torch.zeros(K, 3, dtype=t_chunk.dtype, device=t_chunk.device)
        b_omega = torch.zeros(K, 3, dtype=t_chunk.dtype, device=t_chunk.device)
        b_acc = torch.zeros(K, 3, dtype=t_chunk.dtype, device=t_chunk.device)
        Rot_c_i = torch.zeros(
            K, 3, 3, dtype=t_chunk.dtype, device=t_chunk.device
        )
        t_c_i = torch.zeros(K, 3, dtype=t_chunk.dtype, device=t_chunk.device)

        # Seed first timestep from incoming state
        Rot[0] = state["Rot"]
        v[0] = state["v"]
        p[0] = state["p"]
        b_omega[0] = state["b_omega"]
        b_acc[0] = state["b_acc"]
        Rot_c_i[0] = state["Rot_c_i"]
        t_c_i[0] = state["t_c_i"]
        P = state["P"]

        for i in range(1, K):
            bc_i = (
                bias_corrections_chunk[i]
                if bias_corrections_chunk is not None
                else None
            )
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
                    u_chunk[i],
                    dt_chunk[i - 1],
                    bias_correction=bc_i,
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
                u_chunk[i],
                i,
                measurements_covs_chunk[i],
            )

        traj = (Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i)
        new_state = dict(
            Rot=Rot[-1],
            v=v[-1],
            p=p[-1],
            b_omega=b_omega[-1],
            b_acc=b_acc[-1],
            Rot_c_i=Rot_c_i[-1],
            t_c_i=t_c_i[-1],
            P=P,
        )
        return traj, new_state

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
        """Initialize filter state arrays and covariance."""
        Rot = torch.zeros(N, 3, 3, dtype=dt.dtype, device=dt.device)
        v = torch.zeros(N, 3, dtype=dt.dtype, device=dt.device)
        p = torch.zeros(N, 3, dtype=dt.dtype, device=dt.device)
        b_omega = torch.zeros(N, 3, dtype=dt.dtype, device=dt.device)
        b_acc = torch.zeros(N, 3, dtype=dt.dtype, device=dt.device)
        Rot_c_i = torch.zeros(N, 3, 3, dtype=dt.dtype, device=dt.device)
        t_c_i = torch.zeros(N, 3, dtype=dt.dtype, device=dt.device)

        Rot_c_i[0] = self.Id3
        Rot[0] = self.from_rpy_torch(ang0[0], ang0[1], ang0[2])
        v[0] = v_mes[0]

        P = self.init_covariance()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        """
        Initialize state covariance matrix.

        Uses learned scaling factors when ``initprocesscov_net`` is set.
        """
        P = torch.zeros(
            self.P_dim,
            self.P_dim,
            dtype=self.IdP.dtype,
            device=self.IdP.device,
        )

        if self.initprocesscov_net is not None:
            beta = self.initprocesscov_net.init_cov(self)
        else:
            beta = torch.ones(6, dtype=self.IdP.dtype, device=self.IdP.device)

        P[:2, :2] = self.cov_Rot0 * beta[0] * self.Id2
        P[3:5, 3:5] = self.cov_v0 * beta[1] * self.Id2
        P[9:12, 9:12] = self.cov_b_omega0 * beta[2] * self.Id3
        P[12:15, 12:15] = self.cov_b_acc0 * beta[3] * self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0 * beta[4] * self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0 * beta[5] * self.Id3
        return P

    # ------------------------------------------------------------------
    # Propagation (prediction)
    # ------------------------------------------------------------------

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
        bias_correction=None,
    ):
        """
        Propagate state forward one timestep using classical inertial
        kinematics, optionally with a learned accelerometer bias
        correction Δb_acc(t).

        Args:
            bias_correction: Optional (3,) tensor — additive correction to
                the accelerometer bias for this timestep, predicted by
                LearnedBiasCorrectionNet.
        """
        Rot_prev = Rot_prev.clone()

        # Classical inertial kinematics
        acc_b = u[3:6] - b_acc_prev
        if bias_correction is not None:
            acc_b = acc_b - bias_correction
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

        # Propagate covariance
        P = self.propagate_cov_torch(
            P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt
        )

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov_torch(
        self, P, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u, dt
    ):
        """Propagate covariance matrix one timestep."""
        F = torch.zeros(self.P_dim, self.P_dim, dtype=P.dtype, device=P.device)
        G = torch.zeros(
            self.P_dim, self.Q.shape[0], dtype=P.dtype, device=P.device
        )
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
        Phi = self.IdP + F + 0.5 * F_square + (1.0 / 6.0) * F_square.mm(F)

        # Propagate covariance
        P_new = Phi.mm(P + G.mm(Q).mm(G.t())).mm(Phi.t())

        return P_new

    # ------------------------------------------------------------------
    # Update (correction)
    # ------------------------------------------------------------------

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
        """Update state using zero lateral/vertical velocity constraints."""
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
        """Kalman gain, state correction, and covariance update."""
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
        I_KH = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - K.mm(H)
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

    # ==================================================================
    # Geometry utilities (all device-aware via input tensors)
    # ==================================================================

    @staticmethod
    def skew_torch(x):
        """Skew-symmetric matrix from 3-vector."""
        mat = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
        mat[0, 0] = 0.0
        mat[0, 1] = -x[2]
        mat[0, 2] = x[1]
        mat[1, 0] = x[2]
        mat[1, 1] = 0.0
        mat[1, 2] = -x[0]
        mat[2, 0] = -x[1]
        mat[2, 1] = x[0]
        mat[2, 2] = 0.0
        return mat

    @staticmethod
    def so3exp_torch(phi):
        """SO(3) exponential map."""
        angle = phi.norm()
        Id3 = torch.eye(3, dtype=phi.dtype, device=phi.device)

        if isclose(
            angle, torch.tensor(0.0, dtype=phi.dtype, device=phi.device)
        ):
            skew_phi = TorchIEKF.skew_torch(phi)
            return Id3 + skew_phi

        axis = phi / angle
        skew_axis = TorchIEKF.skew_torch(axis)
        c = angle.cos()
        s = angle.sin()

        return c * Id3 + (1 - c) * TorchIEKF.outer(axis, axis) + s * skew_axis

    @staticmethod
    def se23_exp_torch(xi):
        """SE_2(3) exponential map.  Returns (Rot, x) where x is 3x2."""
        phi = xi[:3]
        angle = torch.norm(phi)
        Id3 = torch.eye(3, dtype=xi.dtype, device=xi.device)

        if isclose(angle, torch.tensor(0.0, dtype=xi.dtype, device=xi.device)):
            skew_phi = TorchIEKF.skew_torch(phi)
            J = Id3 + 0.5 * skew_phi
            Rot = Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = TorchIEKF.skew_torch(axis)
            s = torch.sin(angle)
            c = torch.cos(angle)

            J = (
                (s / angle) * Id3
                + (1 - s / angle) * TorchIEKF.outer(axis, axis)
                + ((1 - c) / angle) * skew_axis
            )
            Rot = (
                c * Id3 + (1 - c) * TorchIEKF.outer(axis, axis) + s * skew_axis
            )

        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    @staticmethod
    def from_rpy_torch(roll, pitch, yaw):
        """Rotation matrix from roll-pitch-yaw (ZYX convention)."""
        return TorchIEKF.rotz_torch(yaw).mm(
            TorchIEKF.roty_torch(pitch).mm(TorchIEKF.rotx_torch(roll))
        )

    @staticmethod
    def rotx_torch(t):
        """Rotation around x-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        M = torch.zeros(3, 3, dtype=t.dtype, device=t.device)
        M[0, 0] = 1.0
        M[0, 1] = 0.0
        M[0, 2] = 0.0
        M[1, 0] = 0.0
        M[1, 1] = c
        M[1, 2] = -s
        M[2, 0] = 0.0
        M[2, 1] = s
        M[2, 2] = c
        return M

    @staticmethod
    def roty_torch(t):
        """Rotation around y-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        M = torch.zeros(3, 3, dtype=t.dtype, device=t.device)
        M[0, 0] = c
        M[0, 1] = 0.0
        M[0, 2] = s
        M[1, 0] = 0.0
        M[1, 1] = 1.0
        M[1, 2] = 0.0
        M[2, 0] = -s
        M[2, 1] = 0.0
        M[2, 2] = c
        return M

    @staticmethod
    def rotz_torch(t):
        """Rotation around z-axis."""
        c = torch.cos(t)
        s = torch.sin(t)
        M = torch.zeros(3, 3, dtype=t.dtype, device=t.device)
        M[0, 0] = c
        M[0, 1] = -s
        M[0, 2] = 0.0
        M[1, 0] = s
        M[1, 1] = c
        M[1, 2] = 0.0
        M[2, 0] = 0.0
        M[2, 1] = 0.0
        M[2, 2] = 1.0
        return M

    @staticmethod
    def outer(a, b):
        """Outer product."""
        return torch.ger(a, b)

    @staticmethod
    def normalize_rot_torch(Rot):
        """Project a near-rotation matrix back onto SO(3) via SVD."""
        U, _, V = torch.svd(Rot)
        S = torch.eye(3, dtype=Rot.dtype, device=Rot.device)
        S[2, 2] = torch.det(U) * torch.det(V)
        return U.mm(S).mm(V.t())

    # ------------------------------------------------------------------
    # Network integration
    # ------------------------------------------------------------------

    def forward_nets(self, u):
        """
        Predict measurement covariances from raw IMU data.

        Args:
            u: IMU measurements (N, 6).

        Returns:
            Measurement covariances (N, 2).
        """
        if self.mes_net is None:
            # Use default covariances if no network
            return self.cov0_measurement.unsqueeze(0).repeat(u.shape[0], 1)

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        measurements_covs = self.mes_net(u_n, self)
        return measurements_covs

    def forward_bias_net(self, u):
        """
        Predict per-timestep accelerometer bias corrections.

        Args:
            u: IMU measurements (N, 6), raw (not normalized).

        Returns:
            Bias corrections (N, 3) in m/s², or None if no
            bias_correction_net is attached.
        """
        if self.bias_correction_net is None:
            return None

        u_n = self.normalize_u(u).t().unsqueeze(0)
        u_n = u_n[:, :6]
        return self.bias_correction_net(u_n, self)

    def set_Q(self):
        """
        Rebuild Q, optionally using learned scaling factors.

        Called once per training step so the computation graph is fresh.
        """
        # Reset to default values
        q_vals = torch.tensor(
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
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.Q.data.copy_(torch.diag(q_vals))

        if self.initprocesscov_net is None:
            return

        # Apply learned scaling (create on same device as current Q buffer)
        beta = self.initprocesscov_net.init_processcov(self)
        Q_new = torch.zeros(
            self.Q.shape[0],
            self.Q.shape[0],
            dtype=self.Q.dtype,
            device=self.Q.device,
        )
        Q_new[:3, :3] = self.cov_omega * beta[0] * self.Id3
        Q_new[3:6, 3:6] = self.cov_acc * beta[1] * self.Id3
        Q_new[6:9, 6:9] = self.cov_b_omega * beta[2] * self.Id3
        Q_new[9:12, 9:12] = self.cov_b_acc * beta[3] * self.Id3
        Q_new[12:15, 12:15] = self.cov_Rot_c_i * beta[4] * self.Id3
        Q_new[15:18, 15:18] = self.cov_t_c_i * beta[5] * self.Id3
        self.Q.data.copy_(Q_new)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_u(self, u):
        """Normalize IMU inputs using dataset statistics."""
        if self.u_loc is None or self.u_std is None:
            return u
        return (u - self.u_loc) / self.u_std

    def get_normalize_u(self, dataset):
        """Load normalization parameters from dataset and move to model device."""
        self.u_loc = dataset.normalize_factors["u_loc"].float().to(self.device)
        self.u_std = dataset.normalize_factors["u_std"].float().to(self.device)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def load(self, args, dataset):
        """
        Load trained network weights (legacy format).

        Args:
            args: Arguments containing path_temp
            dataset: Dataset containing normalization factors
        """
        path_iekf = os.path.join(args.path_temp, "iekfnets.p")
        if os.path.isfile(path_iekf):
            mondict = torch.load(path_iekf, map_location=self.device)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", "green")
        else:
            cprint("IEKF nets NOT loaded", "yellow")
        self.get_normalize_u(dataset)


# Alias for backward compatibility
TORCHIEKF = TorchIEKF
