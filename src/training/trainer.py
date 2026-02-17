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
        self.max_loss = self.training_cfg.get("max_loss")
        self.seed = self.training_cfg.get("seed")

        # Gradient clipping
        gc_cfg = self.training_cfg.get("gradient_clipping")
        self.clip_gradients = gc_cfg.get("enabled")
        self.max_grad_norm = gc_cfg.get("max_norm")

        # Debug
        debug_cfg = self.training_cfg.get("debug", {})
        self.fast_dev_run = debug_cfg.get("fast_dev_run", False)

        # Build components
        self.model = self._build_model()
        self.loss_fn = self._build_loss()
        self.optimizer = build_optimizer(self.optimizer_cfg, self.model)
        self.scheduler = build_scheduler(
            self.optimizer_cfg.get("scheduler", {}), self.optimizer
        )
        self.callbacks = self._build_callbacks()

        # Initialize model with dataset normalization
        self.model.get_normalize_u(dataset)
        self.model.train()

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
            paths_cfg = self.cfg.get("paths", {})
            save_dir = paths_cfg.get("checkpoints", "checkpoints")
            cb_list.add(
                CheckpointCallback(
                    save_dir=save_dir,
                    save_interval=ckpt_cfg.get("save_interval", 1),
                    keep_last_n=ckpt_cfg.get("keep_last_n", 5),
                    save_best=ckpt_cfg.get("save_best", True),
                )
            )

        # WandB logging
        if self.logging_cfg.get("use_wandb", False):
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

            start_time = time.time()
            epoch_metrics = self.train_epoch(epoch)
            elapsed = time.time() - start_time

            epoch_metrics["train/epoch_time"] = elapsed
            self.callbacks.on_epoch_end(self, epoch, epoch_metrics)

            print(
                f"Epoch {epoch:3d} | loss: {epoch_metrics.get('train/loss_epoch', float('nan')):.5f} "
                f"| time: {elapsed:.1f}s"
            )

            # Check early stopping
            for cb in self.callbacks.callbacks:
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    print("Training stopped early.")
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
                loss = self._train_step(dataset_name, Ns)

                step_metrics = {"train/sequence": dataset_name}

                if loss == -1 or (
                    isinstance(loss, torch.Tensor) and torch.isnan(loss)
                ):
                    n_skipped += 1
                    cprint(
                        f"  [step {step}, seq {j}] {dataset_name}: loss invalid",
                        "yellow",
                    )
                    continue
                elif (
                    self.max_loss is not None
                    and isinstance(loss, torch.Tensor)
                    and loss.item() > self.max_loss
                ):
                    n_skipped += 1
                    cprint(
                        f"  [step {step}, seq {j}] {dataset_name}: "
                        f"loss too high ({loss.item():.5f})",
                        "yellow",
                    )
                    continue

                batch_loss = loss if batch_loss is None else batch_loss + loss
                batch_count += 1
                n_sequences += 1
                epoch_loss_sum += loss.item()
                step_metrics["train/loss_step"] = loss.item()
                cprint(
                    f"  [step {step}, seq {j}] {dataset_name}: "
                    f"loss={loss.item():.5f}"
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

            if np.isnan(g_norm) or g_norm > 3 * self.max_grad_norm:
                cprint(f"  gradient norm too large: {g_norm:.5f}", "yellow")
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
        metrics = {
            "train/loss_epoch": epoch_loss / max(n_optimizer_steps, 1),
            "train/loss_sum": epoch_loss_sum,
            "train/grad_norm": avg_grad_norm,
            "train/sequences_skipped": n_skipped,
            "train/n_sequences": n_sequences,
            "train/n_optimizer_steps": n_optimizer_steps,
        }
        for i, pg in enumerate(self.optimizer.param_groups):
            metrics[f"train/lr_group{i}"] = pg["lr"]

        return metrics

    def _train_step(self, dataset_name, Ns):
        """
        Single training step on one sequence.

        Args:
            dataset_name: Name of the training sequence.
            Ns: [start_idx, end_idx] frame range.

        Returns:
            Loss tensor, or -1 if computation failed.
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
        t = t[N0:N_end].double()
        ang_gt = ang_gt[N0:N_end].double()
        p_gt = (p_gt[N0:N_end] - p_gt[N0]).double()
        v_gt = v_gt[N0:N_end].double()
        u = u[N0:N_end].double()

        # Add noise during training
        u = self.dataset.add_noise(u)

        # Rebuild Q each step so the computation graph is not shared
        # across batch boundaries (avoids "backward through graph a second time")
        self.model.set_Q()

        # Forward pass through networks
        measurements_covs = self.model.forward_nets(u)

        # Run filter
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = self.model.run(
            t, u, measurements_covs, v_gt, p_gt, t.shape[0], ang_gt[0]
        )

        # Compute loss
        list_rpe = self.dataset.list_rpe.get(dataset_name)
        if list_rpe is None:
            return -1

        loss = self.loss_fn(Rot, p, None, None, list_rpe=list_rpe, N0=N0)
        return loss

    def _prepare_loss_data(self):
        """Precompute RPE ground truth data for all training sequences."""
        from src.core.torch_iekf import TorchIEKF
        import copy

        # Check for cached data
        list_rpe = {}
        for dataset_name, Ns in self.dataset.datasets_train_filter.items():
            t, ang_gt, p_gt, v_gt, u = self.dataset.get_data(dataset_name)
            end = Ns[1] if Ns[1] is not None else p_gt.shape[0]
            p_gt = p_gt[:end].double()
            ang_gt_sub = ang_gt[:end]

            # Build rotation matrices from Euler angles
            Rot_gt = torch.zeros(end, 3, 3)
            for k in range(end):
                Rot_gt[k] = TorchIEKF.from_rpy_torch(
                    ang_gt_sub[k][0].double(),
                    ang_gt_sub[k][1].double(),
                    ang_gt_sub[k][2].double(),
                ).double()

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
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
