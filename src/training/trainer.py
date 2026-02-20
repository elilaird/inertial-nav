"""
Main training orchestrator for the IEKF with learned covariances.

The Trainer assembles all components (model, dataset, loss, optimizer,
callbacks) from configuration and runs the training loop.
"""

import time
import numpy as np
import torch
from termcolor import cprint

from src.core.torch_iekf import TorchIEKF
from src.losses import get_loss
from src.training.optimizer_factory import build_optimizer, build_scheduler
from src.training.callbacks import (
    CallbackList,
    CheckpointCallback,
    WandBLogger,
    EarlyStopping,
    TestEvalCallback,
)
from src.utils.wandb_utils import log_config


class Trainer:
    """
    Training orchestrator for IEKF with learned covariance networks.

    Assembles model, dataset, loss, optimizer and callbacks from config,
    then runs the training loop with gradient clipping, loss validation,
    and comprehensive logging.

    Args:
        cfg: Full Hydra config (OmegaConf DictConfig or dict).
        dataset: Pre-built dataset instance (BaseIMUDataset).
    """

    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset

        # Extract sub-configs
        self.training_cfg = cfg.get("training")
        self.optimizer_cfg = cfg.get("optimizer")
        self.loss_cfg = cfg.get("loss")
        self.model_cfg = cfg.get("model")
        self.logging_cfg = cfg.get("logging")

        # Experiment metadata
        exp_cfg = cfg.get("experiment", {})
        self.experiment_name = exp_cfg.get("name", "ai-imu-dr")
        self.tags = list(exp_cfg.get("tags", []))

        # Flatten config for WandB
        self.flat_config = log_config(cfg)

        # Training params
        self.epochs = self.training_cfg.get("epochs")
        self.seq_dim = self.training_cfg.get("seq_dim")
        self.batch_size = self.training_cfg.get("batch_size", 9)
        self.steps_per_epoch = self.training_cfg.get("steps_per_epoch", None)
        self.seed = self.training_cfg.get("seed")

        # BPTT config
        bptt_cfg = self.training_cfg.get("bptt", {})
        self.use_bptt = bptt_cfg.get("enabled", False)
        self.bptt_chunk_size = bptt_cfg.get("chunk_size", 500)

        # KL regularization (LatentWorldModel)
        wm_cfg = (
            (self.model_cfg or {}).get("networks", {}).get("world_model", {})
        )
        self.kl_weight = wm_cfg.get("kl_weight", 0.0)
        self.kl_anneal_epochs = wm_cfg.get("kl_anneal_epochs", 50)
        self.current_epoch = 0

        # Gradient clipping
        gc_cfg = self.training_cfg.get("gradient_clipping")
        self.clip_gradients = gc_cfg.get("enabled")
        self.max_grad_norm = gc_cfg.get("max_norm")

        # Debug
        debug_cfg = self.training_cfg.get("debug", {})
        self.fast_dev_run = debug_cfg.get("fast_dev_run", False)

        # Device
        device_str = self.training_cfg.get("device", "auto")
        self.device = self._resolve_device(device_str)
        cprint(f"Using device: {self.device}", "cyan", flush=True)

        # Build components
        self.model = self._build_model()
        self.model.to(self.device)
        cprint(f"\n{self.model}\n", flush=True)
        self.loss_fn = self._build_loss()
        self.optimizer = build_optimizer(self.optimizer_cfg, self.model)
        self.scheduler = build_scheduler(
            self.optimizer_cfg.get("scheduler", {}), self.optimizer
        )
        self.callbacks = self._build_callbacks()

        # Initialize model with dataset normalization
        self.model.get_normalize_u(dataset)
        self.model.train()

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve 'auto' to cuda/cpu, or pass through explicit device strings."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_model(self):
        """Build TorchIEKF model from config."""
        if self.model_cfg:
            return TorchIEKF.build_from_cfg(self.model_cfg)
        return TorchIEKF()

    def _build_loss(self):
        """Build loss function from config."""
        loss_type = self.loss_cfg.get("type", "RPELoss")
        return get_loss(loss_type, self.loss_cfg)

    def _build_callbacks(self):
        """Build callback list from config."""
        cb_list = CallbackList()

        # Checkpointing
        ckpt_cfg = self.training_cfg.get("checkpointing", {})
        if ckpt_cfg.get("enabled", True):
            paths_cfg = self.cfg.get("paths") or {}
            save_dir = paths_cfg.get("checkpoints", "checkpoints")
            cb_list.add(
                CheckpointCallback(
                    save_dir=save_dir,
                    save_interval=ckpt_cfg.get("save_interval", 1),
                    keep_last_n=ckpt_cfg.get("keep_last_n", 5),
                    save_best=ckpt_cfg.get("save_best", True),
                )
            )

        # Test evaluation during training
        validation_cfg = self.training_cfg.get("validation") or {}
        test_eval_cfg = validation_cfg.get("test_eval") or {}
        if test_eval_cfg.get("enabled", False):
            cb_list.add(
                TestEvalCallback(
                    dataset=self.dataset,
                    interval=test_eval_cfg.get("interval", 5),
                )
            )

        # WandB logging
        if self.logging_cfg.get("use_wandb", True):
            wandb_cfg = self.logging_cfg.get("wandb", {})
            cb_list.add(WandBLogger(wandb_cfg, model=self.model))

        # Early stopping
        es_cfg = self.training_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", False):
            cb_list.add(
                EarlyStopping(
                    monitor=es_cfg.get("monitor", "train/loss_epoch"),
                    patience=es_cfg.get("patience", 20),
                )
            )

        return cb_list

    def fit(self):
        """Run the full training loop."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.callbacks.on_train_start(self)

        # Precompute RPE data for loss
        self._prepare_loss_data()

        for epoch in range(1, self.epochs + 1):
            self.callbacks.on_epoch_start(self, epoch)

            self.current_epoch = epoch
            start_time = time.time()
            epoch_metrics = self.train_epoch(epoch)
            elapsed = time.time() - start_time

            epoch_metrics["train/epoch_time"] = elapsed
            self.callbacks.on_epoch_end(self, epoch, epoch_metrics)

            loss_str = f"loss: {epoch_metrics.get('train/loss_epoch', float('nan')):.5f}"
            if "train/base_loss_epoch" in epoch_metrics:
                loss_str += (
                    f" (base: {epoch_metrics['train/base_loss_epoch']:.5f}"
                    f", kl: {epoch_metrics['train/kl_epoch']:.5f})"
                )
            print(
                f"Epoch {epoch:3d} | {loss_str} | time: {elapsed:.1f}s",
                flush=True,
            )

            # Check early stopping
            for cb in self.callbacks.callbacks:
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    print("Training stopped early.", flush=True)
                    self.callbacks.on_train_end(self)
                    return epoch_metrics

            # Fast dev run: only 1 epoch
            if self.fast_dev_run:
                break

            # LR scheduling
            if self.scheduler is not None:
                loss_val = epoch_metrics.get("train/loss_epoch", None)
                if loss_val is not None:
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(loss_val)
                    else:
                        self.scheduler.step()

        self.callbacks.on_train_end(self)
        return epoch_metrics

    def train_epoch(self, epoch):
        """Route to BPTT or standard training for one epoch."""
        if self.use_bptt:
            return self._train_epoch_bptt(epoch)
        return self._train_epoch_standard(epoch)

    def _train_epoch_standard(self, epoch):
        """
        Train one epoch matching the paper's sampling strategy.

        Each optimizer step samples ``batch_size`` sequences WITH REPLACEMENT
        at random offsets, runs the filter, averages the loss, then
        back-propagates and steps. This is repeated ``steps_per_epoch`` times.

        Paper reference: "sample a batch of nine 1-min sequences, where each
        sequence starts at a random arbitrary time."

        Returns:
            Dict of epoch metrics.
        """
        self.model.train()

        train_filter = self.dataset.datasets_train_filter
        seq_names = list(train_filter.keys())
        n_train = len(seq_names)

        # Resolve steps per epoch: default mimics legacy compute budget
        # (one pass through the dataset worth of sequence evaluations)
        steps = self.steps_per_epoch
        if steps is None:
            import math

            steps = max(1, math.ceil(n_train / self.batch_size))
        if self.fast_dev_run:
            steps = 1

        rng = np.random.default_rng(self.seed + epoch)

        epoch_loss = 0.0
        epoch_loss_sum = 0.0  # sum across all valid sequences this epoch
        epoch_losses = {}  # per-component loss sums (e.g. "kl")
        n_sequences = 0
        n_skipped = 0
        n_optimizer_steps = 0
        grad_norms = []

        for step in range(steps):
            # Sample batch_size sequences with replacement
            sampled = rng.choice(seq_names, size=self.batch_size, replace=True)

            batch_loss = None
            batch_count = 0

            for j, dataset_name in enumerate(sampled):
                Ns = train_filter[dataset_name]
                result = self._train_step(dataset_name, Ns)
                loss = result["loss"]

                step_metrics = {"train/sequence": dataset_name}

                if loss == -1 or torch.isnan(loss):
                    n_skipped += 1
                    cprint(
                        f"  [step {step}, seq {j}] {dataset_name}: loss invalid",
                        "yellow",
                        flush=True,
                    )
                    continue

                batch_loss = loss if batch_loss is None else batch_loss + loss
                batch_count += 1
                n_sequences += 1
                epoch_loss_sum += loss.item()
                for k, v in result["losses"].items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                step_metrics["train/loss_step"] = loss.item()
                for k, v in result["losses"].items():
                    if k != "loss" and v > 0:
                        step_metrics[f"train/{k}_step"] = v
                if "kl" in result["losses"] and result["losses"]["kl"] > 0:
                    step_metrics["train/base_loss_step"] = (
                        loss.item() - result["losses"]["kl"]
                    )
                cprint(
                    f"  [step {step}, seq {j}] {dataset_name}: "
                    f"total loss={loss.item():.5f} "
                    + ", ".join(f"{k}={v:.5f}" for k, v in result["losses"].items()),
                    flush=True,
                )

                self.callbacks.on_batch_end(
                    self, epoch, step * self.batch_size + j, step_metrics
                )

            if batch_loss is None or batch_count == 0:
                continue

            # Average loss over the valid sequences in the batch (paper: mean)
            avg_batch_loss = batch_loss / batch_count
            epoch_loss += avg_batch_loss.item()

            avg_batch_loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            if torch.isnan(g_norm):
                cprint(
                    f"  gradient norm is NaN, skipping step",
                    "yellow",
                    flush=True,
                )
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_optimizer_steps += 1

            grad_norms.append(float(g_norm))

            if self.fast_dev_run:
                break

        if n_sequences == 0:
            return {
                "train/loss_epoch": float("nan"),
                "train/sequences_skipped": n_skipped,
            }

        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        avg_epoch_loss = epoch_loss / max(n_optimizer_steps, 1)
        metrics = {
            "train/loss_epoch": avg_epoch_loss,
            "train/loss_sum": epoch_loss_sum,
            "train/grad_norm": avg_grad_norm,
            "train/sequences_skipped": n_skipped,
            "train/n_sequences": n_sequences,
            "train/n_optimizer_steps": n_optimizer_steps,
        }
        denom = max(n_sequences, 1)
        for k, v in epoch_losses.items():
            if k != "loss" and v > 0:
                metrics[f"train/{k}_epoch"] = v / denom
        if "kl" in epoch_losses and epoch_losses["kl"] > 0:
            metrics["train/base_loss_epoch"] = (
                avg_epoch_loss - epoch_losses["kl"] / denom
            )
        for i, pg in enumerate(self.optimizer.param_groups):
            metrics[f"train/lr_group{i}"] = pg["lr"]

        return metrics

    # ------------------------------------------------------------------ #
    # Truncated BPTT training                                              #
    # ------------------------------------------------------------------ #

    def _train_epoch_bptt(self, epoch):
        """
        TBPTT training epoch: each sequence is split into non-overlapping
        chunks of ``bptt_chunk_size`` timesteps.  An optimizer step is taken
        after each chunk and the filter state is detached before the next.

        The same paper-faithful sampling is applied (``batch_size`` sequences
        with replacement per epoch step), and each sequence generates
        ``ceil(seq_dim / chunk_size)`` optimizer steps instead of 1.

        Returns:
            Dict of epoch metrics (same keys as standard mode).
        """
        self.model.train()

        train_filter = self.dataset.datasets_train_filter
        seq_names = list(train_filter.keys())
        n_train = len(seq_names)

        steps = self.steps_per_epoch
        if steps is None:
            import math

            steps = max(1, math.ceil(n_train / self.batch_size))
        if self.fast_dev_run:
            steps = 1

        rng = np.random.default_rng(self.seed + epoch)

        epoch_loss = 0.0
        epoch_losses = {}  # per-component loss sums (e.g. "loss", "kl")
        n_optimizer_steps = 0
        n_sequences = 0
        n_skipped = 0
        grad_norms = []
        sigma_z_epoch = []

        for step in range(steps):
            sampled = rng.choice(seq_names, size=self.batch_size, replace=True)

            for j, dataset_name in enumerate(sampled):
                Ns = train_filter[dataset_name]
                result = self._train_step_bptt(dataset_name, Ns, step, j)

                if result["n_valid"] == 0:
                    n_skipped += 1
                    cprint(
                        f"  [step {step}, seq {j}] {dataset_name}: "
                        f"all chunks invalid/skipped",
                        "yellow",
                        flush=True,
                    )
                    continue

                epoch_loss += result["total_loss"]
                for k, v in result["losses"].items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_optimizer_steps += result["n_valid"]
                n_sequences += 1
                n_skipped += result["n_skipped"]
                grad_norms.extend(result["gnorms"])
                sigma_z_epoch.extend(result["sigma_z"])
                n_valid = result["n_valid"]
                losses_str = ", ".join(
                    f"{k}={v / n_valid:.5f}" for k, v in result["losses"].items()
                )
                cprint(
                    f"  [step {step}, seq {j}] {dataset_name}: "
                    f"total loss={result['total_loss'] / n_valid:.5f} "
                    f"{losses_str} "
                    f"({n_valid} chunks, {result['n_skipped']} skipped)",
                    flush=True,
                )

            if self.fast_dev_run:
                break

        if n_sequences == 0:
            return {
                "train/loss_epoch": float("nan"),
                "train/sequences_skipped": n_skipped,
            }

        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        avg_epoch_loss = epoch_loss / max(n_optimizer_steps, 1)
        metrics = {
            "train/loss_epoch": avg_epoch_loss,
            "train/grad_norm": avg_grad_norm,
            "train/sequences_skipped": n_skipped,
            "train/n_sequences": n_sequences,
            "train/n_optimizer_steps": n_optimizer_steps,
            "train/bptt_chunk_size": self.bptt_chunk_size,
        }
        denom = max(n_optimizer_steps, 1)
        for k, v in epoch_losses.items():
            if k != "loss" and v > 0:
                metrics[f"train/{k}_epoch"] = v / denom
        if "kl" in epoch_losses and epoch_losses["kl"] > 0:
            metrics["train/base_loss_epoch"] = (
                avg_epoch_loss - epoch_losses["kl"] / denom
            )
        if sigma_z_epoch:
            metrics["train/sigma_z"] = float(np.mean(sigma_z_epoch))
        for i, pg in enumerate(self.optimizer.param_groups):
            metrics[f"train/lr_group{i}"] = pg["lr"]
        return metrics

    def _train_step_bptt(self, dataset_name, Ns, step=0, seq_idx=0):
        """
        Run one sequence with Truncated BPTT.

        The sequence is divided into non-overlapping chunks of
        ``self.bptt_chunk_size`` timesteps.  For each chunk:

        1. ``set_Q()`` rebuilds the learned process-noise graph.
        2. ``forward_nets`` runs the measurement-cov CNN on the chunk's IMU.
        3. ``run_chunk`` propagates the filter for ``chunk_size`` timesteps.
        4. RPE loss is computed for pairs fully within this chunk.
        5. ``backward()`` + ``optimizer.step()`` are called.
        6. The filter state is *detached* before the next chunk (the BPTT cut).

        Args:
            dataset_name: Training sequence name.
            Ns:           [start_idx, end_idx] frame range.

        Returns:
            Dict with 'total_loss', 'n_valid', 'gnorms', 'n_skipped',
            'sigma_z', and 'losses' (per-component breakdown).
        """
        from src.core.torch_iekf import TorchIEKF

        # Load + slice (same as _train_step)
        t, ang_gt, p_gt, v_gt, u = self.dataset.get_data(dataset_name)
        t = t[Ns[0] : Ns[1]]
        ang_gt = ang_gt[Ns[0] : Ns[1]]
        p_gt = p_gt[Ns[0] : Ns[1]] - p_gt[Ns[0]]
        v_gt = v_gt[Ns[0] : Ns[1]]
        u = u[Ns[0] : Ns[1]]

        N0, N_end = self._get_start_and_end(self.seq_dim, u)
        t = t[N0:N_end].float().to(self.device)
        ang_gt = ang_gt[N0:N_end].float().to(self.device)
        p_gt = (p_gt[N0:N_end] - p_gt[N0]).float().to(self.device)
        v_gt = v_gt[N0:N_end].float().to(self.device)
        u = self.dataset.add_noise(u[N0:N_end].float().to(self.device))
        N = t.shape[0]

        list_rpe = self.dataset.list_rpe.get(dataset_name)
        if list_rpe is None:
            return {
                "total_loss": 0.0,
                "n_valid": 0,
                "gnorms": [],
                "n_skipped": 1,
                "sigma_z": [],
                "losses": {},
            }

        # Initialise filter state (outside any chunk graph)
        self.model.set_Q()
        state = self.model.init_state(t, u, v_gt, ang_gt[0])
        state = TorchIEKF.detach_state(state)

        totals = {"loss": 0.0, "kl": 0.0}
        n_valid = 0
        n_skipped = 0
        gnorms = []
        sigma_z_vals = []
        chunk_size = self.bptt_chunk_size

        for ci, cs in enumerate(range(0, N, chunk_size)):
            ce = min(cs + chunk_size, N)
            if ce - cs < 2:
                break

            # Fresh graph for this chunk
            self.model.set_Q()

            # CNN runs on chunk only — gradient stays within this backward call
            meas_c = self.model.forward_nets(u[cs:ce])
            bc_c = self.model.forward_bias_net(u[cs:ce])
            gc_c = None
            pns_c = None
            bns_c = None

            # World model overrides / augments standalone networks
            wm_c = self.model.forward_world_model(u[cs:ce])
            if wm_c is not None:
                if wm_c.measurement_covs is not None:
                    meas_c = wm_c.measurement_covs
                if wm_c.acc_bias_corrections is not None:
                    bc_c = wm_c.acc_bias_corrections
                gc_c = wm_c.gyro_bias_corrections
                pns_c = wm_c.process_noise_scaling
                bns_c = wm_c.bias_noise_scaling

            # Filter forward for chunk_size timesteps
            traj, new_state = self.model.run_chunk(
                state,
                t[cs:ce],
                u[cs:ce],
                meas_c,
                bias_corrections_chunk=bc_c,
                gyro_corrections_chunk=gc_c,
                process_noise_scaling_chunk=pns_c,
                bias_noise_scaling_chunk=bns_c,
            )
            Rot_c, _, p_c, *_ = traj

            # RPE loss: pass N0 offset so index math maps into p_c correctly
            loss = self.loss_fn(
                Rot_c,
                p_c,
                None,
                None,
                list_rpe=list_rpe,
                N0=N0 + cs,
            )

            # KL regularization (LatentWorldModel)
            chunk_kl = 0.0
            if (
                self.kl_weight > 0
                and wm_c is not None
                and wm_c.mu_z is not None
            ):
                kl = (
                    -0.5
                    * (
                        1
                        + wm_c.log_var_z
                        - wm_c.mu_z.pow(2)
                        - wm_c.log_var_z.exp()
                    ).mean()
                )
                # anneal = min(
                #     1.0, self.current_epoch / max(1, self.kl_anneal_epochs)
                # )
                anneal = 1.0
                chunk_kl = (self.kl_weight * anneal * kl).item()
                loss = loss + self.kl_weight * anneal * kl

            if loss == -1 or torch.isnan(loss):
                n_skipped += 1
                cprint(
                    f"  [step {step}, seq {seq_idx}, chunk {ci}] "
                    f"{dataset_name}: loss invalid",
                    "yellow",
                    flush=True,
                )
                state = TorchIEKF.detach_state(new_state)
                continue

            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            if torch.isnan(g_norm):
                cprint(
                    f"  [bptt chunk {ci}] gradient norm is NaN, skipping step",
                    "yellow",
                )
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_valid += 1
                totals["loss"] += loss.item()
                totals["kl"] += chunk_kl
                gnorms.append(float(g_norm))
                if wm_c is not None and wm_c.log_var_z is not None:
                    sigma_z_vals.append(
                        (0.5 * wm_c.log_var_z).exp().mean().item()
                    )

            # ---- BPTT cut: detach state before next chunk ----
            state = TorchIEKF.detach_state(new_state)

            if self.fast_dev_run:
                break

        return {
            "total_loss": totals["loss"],
            "n_valid": n_valid,
            "gnorms": gnorms,
            "n_skipped": n_skipped,
            "sigma_z": sigma_z_vals,
            "losses": totals,
        }

    def _train_step(self, dataset_name, Ns):
        """
        Single training step on one sequence.

        Args:
            dataset_name: Name of the training sequence.
            Ns: [start_idx, end_idx] frame range.

        Returns:
            Dict with 'loss' (tensor or -1) and 'losses' (component breakdown).
        """
        # Load and prepare data
        t, ang_gt, p_gt, v_gt, u = self.dataset.get_data(dataset_name)
        t = t[Ns[0] : Ns[1]]
        ang_gt = ang_gt[Ns[0] : Ns[1]]
        p_gt = p_gt[Ns[0] : Ns[1]] - p_gt[Ns[0]]
        v_gt = v_gt[Ns[0] : Ns[1]]
        u = u[Ns[0] : Ns[1]]

        # Random subsequence sampling
        N0, N_end = self._get_start_and_end(self.seq_dim, u)
        t = t[N0:N_end].float().to(self.device)
        ang_gt = ang_gt[N0:N_end].float().to(self.device)
        p_gt = (p_gt[N0:N_end] - p_gt[N0]).float().to(self.device)
        v_gt = v_gt[N0:N_end].float().to(self.device)
        u = u[N0:N_end].float().to(self.device)

        # Add noise during training
        u = self.dataset.add_noise(u)

        # Rebuild Q each step so the computation graph is not shared
        # across batch boundaries (avoids "backward through graph a second time")
        self.model.set_Q()

        # Forward pass through networks
        measurements_covs = self.model.forward_nets(u)
        bias_corrections = self.model.forward_bias_net(u)
        gyro_corrections = None
        process_noise_scaling = None

        # World model overrides / augments standalone networks
        bias_noise_scaling = None
        wm_out = self.model.forward_world_model(u)
        if wm_out is not None:
            if wm_out.measurement_covs is not None:
                measurements_covs = wm_out.measurement_covs
            if wm_out.acc_bias_corrections is not None:
                bias_corrections = wm_out.acc_bias_corrections
            gyro_corrections = wm_out.gyro_bias_corrections
            process_noise_scaling = wm_out.process_noise_scaling
            bias_noise_scaling = wm_out.bias_noise_scaling

        # Run filter
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.model.run(
            t,
            u,
            measurements_covs,
            v_gt,
            p_gt,
            t.shape[0],
            ang_gt[0],
            bias_corrections=bias_corrections,
            gyro_corrections=gyro_corrections,
            process_noise_scaling=process_noise_scaling,
            bias_noise_scaling=bias_noise_scaling,
        )

        # Compute loss
        list_rpe = self.dataset.list_rpe.get(dataset_name)
        if list_rpe is None:
            return {"loss": -1, "losses": {}}

        loss = self.loss_fn(Rot, p, None, None, list_rpe=list_rpe, N0=N0)

        # KL regularization (LatentWorldModel)
        losses = {}
        if (
            isinstance(loss, torch.Tensor)
            and self.kl_weight > 0
            and wm_out is not None
            and wm_out.mu_z is not None
        ):
            kl = (
                -0.5
                * (
                    1
                    + wm_out.log_var_z
                    - wm_out.mu_z.pow(2)
                    - wm_out.log_var_z.exp()
                ).mean()
            )
            # anneal = min(
            #     1.0, self.current_epoch / max(1, self.kl_anneal_epochs)
            # )
            anneal = 1.0
            losses["kl"] = (self.kl_weight * anneal * kl).item()
            loss = loss + self.kl_weight * anneal * kl

        return {"loss": loss, "losses": losses}

    def _prepare_loss_data(self):
        """Precompute RPE ground truth data for all training sequences."""
        from src.core.torch_iekf import TorchIEKF
        import copy

        # Check for cached data
        list_rpe = {}
        for dataset_name, Ns in self.dataset.datasets_train_filter.items():
            t, ang_gt, p_gt, v_gt, u = self.dataset.get_data(dataset_name)
            end = Ns[1] if Ns[1] is not None else p_gt.shape[0]
            p_gt = p_gt[:end].float().to(self.device)
            ang_gt_sub = ang_gt[:end]

            # Build rotation matrices from Euler angles
            Rot_gt = torch.zeros(
                end, 3, 3, dtype=torch.float32, device=self.device
            )
            for k in range(end):
                Rot_gt[k] = TorchIEKF.from_rpy_torch(
                    ang_gt_sub[k][0].float(),
                    ang_gt_sub[k][1].float(),
                    ang_gt_sub[k][2].float(),
                ).to(self.device)

            rpe_data = self.loss_fn.precompute(Rot_gt, p_gt)
            if len(rpe_data[0]) > 0:
                list_rpe[dataset_name] = rpe_data
            else:
                cprint(
                    f"{dataset_name} has no valid RPE pairs, removing",
                    "yellow",
                )

        self.dataset.list_rpe = list_rpe

        # Remove sequences with no valid RPE data
        for name in list(self.dataset.datasets_train_filter.keys()):
            if name not in list_rpe:
                self.dataset.datasets_train_filter.pop(name)

    @staticmethod
    def _get_start_and_end(seq_dim, u):
        """Get random start and end indices for subsequence sampling."""
        if seq_dim is None or u.shape[0] <= seq_dim:
            return 0, u.shape[0]
        N0 = 10 * int(np.random.randint(0, (u.shape[0] - seq_dim) / 10))
        return N0, N0 + seq_dim

    def save_model(self, path):
        """Save model state dict (legacy format compatible)."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state dict."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
