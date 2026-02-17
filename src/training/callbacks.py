"""
Training callback system.

Callbacks hook into the training loop at defined points (epoch start/end,
batch end) to implement cross-cutting concerns like checkpointing,
logging, and early stopping.
"""

import os
import torch
from termcolor import cprint


class Callback:
    """Base callback with no-op hooks."""

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_batch_end(self, trainer, epoch, batch_idx, metrics):
        pass


class CallbackList:
    """Manages a collection of callbacks."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def add(self, callback):
        self.callbacks.append(callback)

    def on_train_start(self, trainer):
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer, epoch):
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer, epoch, metrics):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_end(self, trainer, epoch, batch_idx, metrics):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, epoch, batch_idx, metrics)


class CheckpointCallback(Callback):
    """
    Save model checkpoints periodically and track the best model.

    Args:
        save_dir: Directory for checkpoint files.
        save_interval: Save every N epochs.
        keep_last_n: Keep only the last N checkpoints (0 = keep all).
        save_best: Whether to track and save the best model.
        monitor: Metric name to monitor for best model (default: "train/loss_epoch").
    """

    def __init__(
        self,
        save_dir,
        save_interval=1,
        keep_last_n=5,
        save_best=True,
        monitor="train/loss_epoch",
    ):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.monitor = monitor
        self.best_metric = float("inf")
        self.saved_checkpoints = []

    def on_train_start(self, trainer):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch, metrics):
        # Periodic save
        if epoch % self.save_interval == 0:
            path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
            self._save_checkpoint(trainer, path, epoch, metrics)
            self.saved_checkpoints.append(path)

            # Prune old checkpoints
            if self.keep_last_n > 0:
                while len(self.saved_checkpoints) > self.keep_last_n:
                    old_path = self.saved_checkpoints.pop(0)
                    if os.path.exists(old_path) and "best" not in old_path:
                        os.remove(old_path)

        # Best model tracking
        if self.save_best and self.monitor in metrics:
            current = metrics[self.monitor]
            if current < self.best_metric:
                self.best_metric = current
                path = os.path.join(self.save_dir, "best.pth")
                self._save_checkpoint(trainer, path, epoch, metrics)
                cprint(
                    f"New best model (epoch {epoch}, "
                    f"{self.monitor}={current:.6f})",
                    "green",
                )

    @staticmethod
    def _save_checkpoint(trainer, path, epoch, metrics):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "metrics": metrics,
        }
        if trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path, model, optimizer=None):
        """Load a checkpoint into model and optionally optimizer."""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


class WandBLogger(Callback):
    """
    Log metrics, gradients, and artifacts to Weights & Biases.

    Args:
        cfg: WandB config section from main config.
        model: Model to watch for gradient logging.
    """

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        self.run = None
        self._batch_log_freq = cfg.get("log_freq", {}).get("batch", 10)
        self._gradient_log_freq = cfg.get("log_freq", {}).get("gradient", 100)

    def on_train_start(self, trainer):
        try:
            import wandb
        except ImportError:
            cprint("wandb not installed, skipping WandB logging", "yellow")
            return

        SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
        self.run = wandb.init(
            project=self.cfg.get("project", "ai-imu-dr"),
            entity=self.cfg.get("entity", None),
            name=f"{trainer.experiment_name}_{SLURM_JOB_ID}",
            config=trainer.flat_config,
            mode=self.cfg.get("mode", "online"),
            save_code=self.cfg.get("save_code", False),
            tags=trainer.tags,
        )

        if self.model is not None:
            wandb.watch(
                self.model, log="all", log_freq=self._gradient_log_freq
            )

    def on_train_end(self, trainer):
        if self.run is not None:
            import wandb

            wandb.finish()

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.run is None:
            return
        import wandb

        log_dict = {k: v for k, v in metrics.items()}
        log_dict["epoch"] = epoch
        wandb.log(log_dict, step=epoch)

    def on_batch_end(self, trainer, epoch, batch_idx, metrics):
        if self.run is None:
            return
        if batch_idx % self._batch_log_freq != 0:
            return
        import wandb

        wandb.log(metrics)


class EarlyStopping(Callback):
    """
    Stop training if monitored metric stops improving.

    Args:
        monitor: Metric name to watch.
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, monitor="train/loss_epoch", patience=20, min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                cprint(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)",
                    "yellow",
                )
