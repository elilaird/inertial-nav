"""Tests for training infrastructure (Phase 4)."""

import pytest
import torch
import os
import sys
import tempfile
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.torch_iekf import TorchIEKF
from src.models.init_process_cov_net import InitProcessCovNet
from src.models.measurement_cov_net import MeasurementCovNet
from src.training.optimizer_factory import build_optimizer, build_scheduler
from src.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStopping,
    WandBLogger,
)


# ==================== Optimizer Factory Tests ====================


class TestOptimizerFactory:
    def setup_method(self):
        self.model = TorchIEKF()
        self.model.initprocesscov_net = InitProcessCovNet()
        self.model.mes_net = MeasurementCovNet(input_channels=6, output_dim=2)

    def test_build_adam(self):
        cfg = {
            "type": "Adam",
            "param_groups": {
                "init_process_cov_net": {"lr": 1e-3},
                "measurement_cov_net": {"lr": 1e-4},
            },
        }
        opt = build_optimizer(cfg, self.model)
        assert isinstance(opt, torch.optim.Adam)
        assert len(opt.param_groups) == 2

    def test_build_adamw(self):
        cfg = {
            "type": "AdamW",
            "param_groups": {
                "init_process_cov_net": {"lr": 1e-3},
            },
        }
        opt = build_optimizer(cfg, self.model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_build_sgd(self):
        cfg = {
            "type": "SGD",
            "lr": 1e-3,
            "momentum": 0.9,
            "param_groups": {
                "init_process_cov_net": {"lr": 1e-3},
            },
        }
        opt = build_optimizer(cfg, self.model)
        assert isinstance(opt, torch.optim.SGD)

    def test_per_submodule_lr(self):
        cfg = {
            "type": "Adam",
            "param_groups": {
                "measurement_cov_net": {
                    "cov_net": {"lr": 1e-3, "weight_decay": 1e-8},
                    "cov_lin": {"lr": 1e-4, "weight_decay": 1e-8},
                }
            },
        }
        opt = build_optimizer(cfg, self.model)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 1e-3
        assert opt.param_groups[1]["lr"] == 1e-4

    def test_fallback_no_networks(self):
        model = TorchIEKF()  # No networks
        cfg = {
            "type": "Adam",
            "param_groups": {
                "init_process_cov_net": {"lr": 1e-3},
            },
        }
        opt = build_optimizer(cfg, model)
        # Should fallback to all params
        assert len(opt.param_groups) >= 1

    def test_unknown_optimizer_raises(self):
        cfg = {"type": "Unknown"}
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(cfg, self.model)


class TestSchedulerFactory:
    def setup_method(self):
        self.model = TorchIEKF()
        self.model.initprocesscov_net = InitProcessCovNet()
        opt_cfg = {
            "type": "Adam",
            "param_groups": {"init_process_cov_net": {"lr": 1e-3}},
        }
        self.optimizer = build_optimizer(opt_cfg, self.model)

    def test_disabled_returns_none(self):
        sched = build_scheduler({"enabled": False}, self.optimizer)
        assert sched is None

    def test_reduce_lr_on_plateau(self):
        cfg = {
            "enabled": True,
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 5,
        }
        sched = build_scheduler(cfg, self.optimizer)
        assert sched is not None

    def test_step_lr(self):
        cfg = {"enabled": True, "type": "StepLR", "step_size": 10}
        sched = build_scheduler(cfg, self.optimizer)
        assert sched is not None

    def test_cosine_annealing(self):
        cfg = {"enabled": True, "type": "CosineAnnealingLR", "T_max": 50}
        sched = build_scheduler(cfg, self.optimizer)
        assert sched is not None

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(
                {"enabled": True, "type": "Unknown"}, self.optimizer
            )


# ==================== Callback Tests ====================


class TestCallbackList:
    def test_add_and_call(self):
        called = []

        class TrackerCallback(Callback):
            def on_epoch_start(self, trainer, epoch):
                called.append(("start", epoch))

            def on_epoch_end(self, trainer, epoch, metrics):
                called.append(("end", epoch))

        cb_list = CallbackList()
        cb_list.add(TrackerCallback())
        cb_list.on_epoch_start(None, 1)
        cb_list.on_epoch_end(None, 1, {})

        assert len(called) == 2
        assert called[0] == ("start", 1)
        assert called[1] == ("end", 1)


class TestCheckpointCallback:
    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        model = TorchIEKF()
        model.initprocesscov_net = InitProcessCovNet()

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Simulate trainer
        class FakeTrainer:
            pass

        trainer = FakeTrainer()
        trainer.model = model
        trainer.optimizer = opt

        cb = CheckpointCallback(save_dir=tmpdir, save_interval=1)
        cb.on_train_start(trainer)
        cb.on_epoch_end(trainer, 1, {"train/loss_epoch": 1.0})

        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch1.pth")
        assert os.path.exists(ckpt_path)

        # Load checkpoint
        model2 = TorchIEKF()
        model2.initprocesscov_net = InitProcessCovNet()
        epoch, metrics = CheckpointCallback.load_checkpoint(ckpt_path, model2)
        assert epoch == 1

    def test_best_model_tracking(self):
        tmpdir = tempfile.mkdtemp()
        model = TorchIEKF()

        class FakeTrainer:
            pass

        trainer = FakeTrainer()
        trainer.model = model
        trainer.optimizer = None

        cb = CheckpointCallback(
            save_dir=tmpdir, save_best=True, monitor="train/loss_epoch"
        )
        cb.on_train_start(trainer)

        cb.on_epoch_end(trainer, 1, {"train/loss_epoch": 1.0})
        assert cb.best_metric == 1.0

        cb.on_epoch_end(trainer, 2, {"train/loss_epoch": 0.5})
        assert cb.best_metric == 0.5
        assert os.path.exists(os.path.join(tmpdir, "best.pth"))

    def test_keep_last_n(self):
        tmpdir = tempfile.mkdtemp()
        model = TorchIEKF()

        class FakeTrainer:
            pass

        trainer = FakeTrainer()
        trainer.model = model
        trainer.optimizer = None

        cb = CheckpointCallback(
            save_dir=tmpdir, save_interval=1, keep_last_n=2, save_best=False
        )
        cb.on_train_start(trainer)

        for epoch in range(1, 5):
            cb.on_epoch_end(trainer, epoch, {})

        # Should only keep last 2
        assert not os.path.exists(
            os.path.join(tmpdir, "checkpoint_epoch1.pth")
        )
        assert not os.path.exists(
            os.path.join(tmpdir, "checkpoint_epoch2.pth")
        )
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_epoch3.pth"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_epoch4.pth"))


class TestEarlyStopping:
    def test_no_stop_while_improving(self):
        es = EarlyStopping(monitor="loss", patience=3)
        es.on_epoch_end(None, 1, {"loss": 1.0})
        es.on_epoch_end(None, 2, {"loss": 0.8})
        es.on_epoch_end(None, 3, {"loss": 0.6})
        assert not es.should_stop

    def test_stop_after_patience(self):
        es = EarlyStopping(monitor="loss", patience=3)
        es.on_epoch_end(None, 1, {"loss": 0.5})
        es.on_epoch_end(None, 2, {"loss": 0.6})
        es.on_epoch_end(None, 3, {"loss": 0.7})
        assert not es.should_stop
        es.on_epoch_end(None, 4, {"loss": 0.8})
        assert es.should_stop

    def test_reset_on_improvement(self):
        es = EarlyStopping(monitor="loss", patience=2)
        es.on_epoch_end(None, 1, {"loss": 1.0})
        es.on_epoch_end(None, 2, {"loss": 1.1})
        assert not es.should_stop
        es.on_epoch_end(None, 3, {"loss": 0.5})  # improvement
        es.on_epoch_end(None, 4, {"loss": 0.6})
        assert not es.should_stop  # wait reset

    def test_missing_metric_ignored(self):
        es = EarlyStopping(monitor="loss", patience=2)
        es.on_epoch_end(None, 1, {"other": 1.0})
        assert not es.should_stop


# ==================== Trainer Component Tests ====================


class TestTrainerComponents:
    def _make_fake_dataset(self, tmpdir):
        """Create a minimal fake dataset for Trainer testing."""
        data_dir = os.path.join(tmpdir, "data")
        temp_dir = os.path.join(tmpdir, "temp")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Create a fake sequence with enough data for RPE
        N = 5000
        t = torch.linspace(0, N * 0.01, N).float()
        u = torch.randn(N, 6).float()
        ang_gt = torch.zeros(N, 3).float()
        p_gt = torch.zeros(N, 3).float()
        p_gt[:, 0] = torch.linspace(0, 500, N)  # 500m straight line
        v_gt = torch.zeros(N, 3).float()
        v_gt[:, 0] = 10.0

        mondict = {
            "t": t,
            "u": u,
            "ang_gt": ang_gt,
            "p_gt": p_gt,
            "v_gt": v_gt,
            "name": "fake_seq",
        }
        with open(os.path.join(data_dir, "fake_seq.p"), "wb") as f:
            pickle.dump(mondict, f)

        # Build dataset using our ConcreteDataset from test_datasets
        from tests.test_datasets import ConcreteDataset

        cfg = {
            "path_data_save": data_dir,
            "path_results": os.path.join(tmpdir, "results"),
            "path_temp": temp_dir,
        }
        ds = ConcreteDataset(cfg)
        ds.datasets_train_filter["fake_seq"] = [0, 5000]
        return ds

    def test_trainer_init(self):
        tmpdir = tempfile.mkdtemp()
        dataset = self._make_fake_dataset(tmpdir)

        cfg = {
            "training": {
                "epochs": 1,
                "seq_dim": 2000,
                "seed": 42,
                "gradient_clipping": {"enabled": True, "max_norm": 1.0},
                "checkpointing": {"enabled": False},
                "debug": {"fast_dev_run": True},
            },
            "optimizer": {
                "type": "Adam",
                "param_groups": {},
                "scheduler": {"enabled": False},
            },
            "loss": {"type": "RPELoss"},
            "model": {"networks": {}},
            "logging": {"use_wandb": False},
            "experiment": {"name": "test", "tags": []},
        }

        from src.training.trainer import Trainer

        trainer = Trainer(cfg, dataset)
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None

    def test_trainer_fit_fast_dev_run(self):
        tmpdir = tempfile.mkdtemp()
        dataset = self._make_fake_dataset(tmpdir)

        cfg = {
            "training": {
                "epochs": 1,
                "seq_dim": 2000,
                "seed": 42,
                "gradient_clipping": {"enabled": True, "max_norm": 1.0},
                "checkpointing": {"enabled": False},
                "debug": {"fast_dev_run": True},
            },
            "optimizer": {
                "type": "Adam",
                "param_groups": {
                    "measurement_cov_net": {"lr": 1e-4},
                },
                "scheduler": {"enabled": False},
            },
            "loss": {"type": "RPELoss"},
            "model": {
                "networks": {
                    "measurement_cov": {
                        "enabled": True,
                        "type": "MeasurementCovNet",
                        "architecture": {"input_channels": 6, "output_dim": 2},
                    },
                }
            },
            "logging": {"use_wandb": False},
            "experiment": {"name": "test", "tags": []},
        }

        from src.training.trainer import Trainer

        trainer = Trainer(cfg, dataset)
        metrics = trainer.fit()
        assert "train/loss_epoch" in metrics

    def test_trainer_with_networks(self):
        tmpdir = tempfile.mkdtemp()
        dataset = self._make_fake_dataset(tmpdir)

        cfg = {
            "training": {
                "epochs": 1,
                "seq_dim": 2000,
                "seed": 42,
                "gradient_clipping": {"enabled": True, "max_norm": 1.0},
                "checkpointing": {"enabled": False},
                "debug": {"fast_dev_run": True},
            },
            "optimizer": {
                "type": "Adam",
                "param_groups": {
                    "init_process_cov_net": {"lr": 1e-4},
                    "measurement_cov_net": {"lr": 1e-4},
                },
                "scheduler": {"enabled": False},
            },
            "loss": {"type": "RPELoss"},
            "model": {
                "networks": {
                    "init_process_cov": {
                        "enabled": True,
                        "type": "InitProcessCovNet",
                        "architecture": {"output_dim": 6},
                    },
                    "measurement_cov": {
                        "enabled": True,
                        "type": "MeasurementCovNet",
                        "architecture": {"input_channels": 6, "output_dim": 2},
                    },
                }
            },
            "logging": {"use_wandb": False},
            "experiment": {"name": "test", "tags": []},
        }

        from src.training.trainer import Trainer

        trainer = Trainer(cfg, dataset)
        assert trainer.model.initprocesscov_net is not None
        assert trainer.model.mes_net is not None
        metrics = trainer.fit()
        assert "train/loss_epoch" in metrics

    def test_trainer_save_load_model(self):
        tmpdir = tempfile.mkdtemp()
        dataset = self._make_fake_dataset(tmpdir)

        cfg = {
            "training": {
                "epochs": 1,
                "seq_dim": 2000,
                "seed": 42,
                "gradient_clipping": {"enabled": True, "max_norm": 1.0},
                "checkpointing": {"enabled": False},
                "debug": {"fast_dev_run": True},
            },
            "optimizer": {
                "type": "Adam",
                "param_groups": {},
                "scheduler": {"enabled": False},
            },
            "loss": {"type": "RPELoss"},
            "model": {"networks": {}},
            "logging": {"use_wandb": False},
            "experiment": {"name": "test", "tags": []},
        }

        from src.training.trainer import Trainer

        trainer = Trainer(cfg, dataset)

        # Save and reload
        save_path = os.path.join(tmpdir, "model.pth")
        trainer.save_model(save_path)
        assert os.path.exists(save_path)

        trainer.load_model(save_path)  # Should not raise

    def _make_multi_seq_dataset(self, tmpdir, n_seqs=4):
        """Create a fake dataset with multiple sequences for batch testing."""
        data_dir = os.path.join(tmpdir, "data")
        temp_dir = os.path.join(tmpdir, "temp")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        from tests.test_datasets import ConcreteDataset

        for s in range(n_seqs):
            N = 5000
            t = torch.linspace(0, N * 0.01, N).float()
            u = torch.randn(N, 6).float()
            ang_gt = torch.zeros(N, 3).float()
            p_gt = torch.zeros(N, 3).float()
            p_gt[:, 0] = torch.linspace(0, 500, N)
            v_gt = torch.zeros(N, 3).float()
            v_gt[:, 0] = 10.0

            name = f"fake_seq_{s}"
            mondict = {
                "t": t,
                "u": u,
                "ang_gt": ang_gt,
                "p_gt": p_gt,
                "v_gt": v_gt,
                "name": name,
            }
            with open(os.path.join(data_dir, f"{name}.p"), "wb") as f:
                pickle.dump(mondict, f)

        cfg = {
            "path_data_save": data_dir,
            "path_results": os.path.join(tmpdir, "results"),
            "path_temp": temp_dir,
        }
        ds = ConcreteDataset(cfg)
        for s in range(n_seqs):
            ds.datasets_train_filter[f"fake_seq_{s}"] = [0, 5000]
        return ds

    def test_trainer_batch_size_larger_than_seqs(self):
        """batch_size > n_sequences: all sequences in one batch (legacy behavior)."""
        tmpdir = tempfile.mkdtemp()
        dataset = self._make_multi_seq_dataset(tmpdir, n_seqs=3)

        cfg = {
            "training": {
                "epochs": 1,
                "seq_dim": 2000,
                "batch_size": 10,
                "seed": 42,
                "gradient_clipping": {"enabled": True, "max_norm": 1.0},
                "checkpointing": {"enabled": False},
                "debug": {"fast_dev_run": False},
            },
            "optimizer": {
                "type": "Adam",
                "param_groups": {
                    "measurement_cov_net": {"lr": 1e-4},
                },
                "scheduler": {"enabled": False},
            },
            "loss": {"type": "RPELoss"},
            "model": {
                "networks": {
                    "measurement_cov": {
                        "enabled": True,
                        "type": "MeasurementCovNet",
                        "architecture": {"input_channels": 6, "output_dim": 2},
                    },
                }
            },
            "logging": {"use_wandb": False},
            "experiment": {"name": "test", "tags": []},
        }

        from src.training.trainer import Trainer

        trainer = Trainer(cfg, dataset)
        metrics = trainer.fit()
        assert "train/loss_epoch" in metrics
        # 3 sequences, batch_size=10 → remainder flush = 1 optimizer step
        assert metrics["train/n_optimizer_steps"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
