# Validation Test Plan

Specific tests to run to validate training, data pipeline, and evaluation components.

---

## 1. Unit Tests (No Data Required)

Run these first. They use synthetic data and should all pass without any KITTI data.

### Full Suite

```bash
cd /Users/eli/Projects/inertial-nav/.worktrees/phase1-modernization
source .venv/bin/activate
python -m pytest tests/ -v
```

Expected: 173 tests passing.

### By Component

**Data pipeline** (21 tests) — dataset loading, transforms, normalization, noise injection:
```bash
python -m pytest tests/test_datasets.py -v
```

**Loss functions** (19 tests) — RPE/ATE loss computation, gradient flow, precomputation:
```bash
python -m pytest tests/test_losses.py -v
```

**Training infrastructure** (23 tests) — optimizer factory, scheduler, callbacks, Trainer init/fit:
```bash
python -m pytest tests/test_trainer.py -v
```

**Evaluation metrics & visualization** (20 tests) — RPE/ATE/orientation metrics, all plot functions:
```bash
python -m pytest tests/test_evaluation.py -v
```

---

## 2. Data Pipeline Smoke Tests (Requires KITTI Data)

These test the full data flow from raw KITTI → pickle → dataset loading.

### 2a. Verify Pickle Data Exists

Check that preprocessed pickle files are available:
```bash
ls data/*.p
```

Expected: One `.p` file per KITTI sequence (e.g., `2011_10_03_drive_0042_extract.p`). If missing, run data preparation first:
```bash
python scripts/prepare_kitti_data.py \
  --kitti_raw /path/to/KITTI/raw \
  --output data/
```

### 2b. Load a Single Sequence

Verify a sequence loads correctly and has expected shapes:
```python
python -c "
from src.data.kitti_dataset import KITTIDataset
cfg = {
    'path_data_save': 'data',
    'path_results': 'results',
    'path_temp': 'temp',
}
ds = KITTIDataset(cfg, split='train')
print(f'Sequences: {len(ds)}')
print(f'Train: {ds.datasets_train}')
print(f'Test: {ds.datasets_test}')

# Load one sequence
name = ds.datasets_train[0]
t, ang_gt, p_gt, v_gt, u = ds.get_data(name)
print(f'\nSequence: {name}')
print(f'  t:      {t.shape}  range=[{t[0]:.2f}, {t[-1]:.2f}]')
print(f'  u:      {u.shape}  (IMU: gyro+acc)')
print(f'  p_gt:   {p_gt.shape}  distance={float((p_gt[-1]-p_gt[0]).norm()):.1f}m')
print(f'  v_gt:   {v_gt.shape}')
print(f'  ang_gt: {ang_gt.shape}')
"
```

Expected output: Shapes `(N, 6)` for IMU, `(N, 3)` for positions/velocities/angles, with `N` in the thousands.

### 2c. Test Normalization

Verify normalization factors are computed and cached:
```python
python -c "
from src.data.kitti_dataset import KITTIDataset
import torch
cfg = {
    'path_data_save': 'data',
    'path_results': 'results',
    'path_temp': 'temp',
}
ds = KITTIDataset(cfg, split='train')
nf = ds.normalize_factors
print(f'u_loc: {nf[\"u_loc\"]}')
print(f'u_std: {nf[\"u_std\"]}')
print(f'Cached to: temp/normalization.p')
"
```

### 2d. Test Transforms Compose

```python
python -c "
from src.data.transforms import Compose, AddIMUNoise, RandomSubsequenceSampler
from src.data.kitti_dataset import KITTIDataset
cfg = {
    'path_data_save': 'data',
    'path_results': 'results',
    'path_temp': 'temp',
}
ds = KITTIDataset(cfg, split='train')
t, ang_gt, p_gt, v_gt, u = ds.get_data(ds.datasets_train[0])

transform = Compose([
    RandomSubsequenceSampler(seq_dim=6000),
    AddIMUNoise(sigma_gyro=1e-4, sigma_acc=1e-3),
])
data = {'t': t, 'u': u, 'ang_gt': ang_gt, 'p_gt': p_gt, 'v_gt': v_gt}
out = transform(data)
print(f'Original length: {u.shape[0]}')
print(f'After transforms: {out[\"u\"].shape[0]}')
print(f'Subsequence start: N0={out[\"N0\"]}')
"
```

---

## 3. Training Smoke Tests

### 3a. Fast Dev Run (No Real Data Needed)

Uses synthetic data via the test infrastructure. Verifies the full training loop (model build, loss, optimizer, forward pass, backward pass, gradient clipping):
```bash
python -m pytest tests/test_trainer.py::TestTrainerComponents::test_trainer_fit_fast_dev_run -v
```

### 3b. Fast Dev Run with Networks

Same but with InitProcessCovNet and MeasurementCovNet enabled:
```bash
python -m pytest tests/test_trainer.py::TestTrainerComponents::test_trainer_with_networks -v
```

### 3c. Hydra Training Entry Point (Requires KITTI Data)

Run 1 epoch with real data, WandB disabled:
```bash
python src/train.py \
  training.epochs=1 \
  training.debug.fast_dev_run=true \
  logging.use_wandb=false \
  training.checkpointing.enabled=false
```

What to check:
- Config prints without errors
- Loss is a finite number (not NaN, not huge)
- No import errors or shape mismatches
- Completes without crashing

### 3d. Short Training Run (Requires KITTI Data)

Run 3 epochs to verify loss decreases:
```bash
python src/train.py \
  training.epochs=3 \
  logging.use_wandb=false \
  training.checkpointing.enabled=true \
  training.checkpointing.save_interval=1 \
  training.checkpointing.keep_last_n=2
```

What to check:
- Loss printed each epoch
- Loss decreases (or at least doesn't explode)
- Checkpoints created: `ls checkpoints/*.pth`
- Only last 2 checkpoints kept

### 3e. Training with Different Model Configs

Test that model config swapping works:
```bash
# Default: learned covariances
python src/train.py training.epochs=1 training.debug.fast_dev_run=true \
  logging.use_wandb=false model=iekf_learned_cov

# With learned dynamics
python src/train.py training.epochs=1 training.debug.fast_dev_run=true \
  logging.use_wandb=false model=iekf_learned_dynamics
```

### 3f. Training with WandB (Requires KITTI Data + WandB Account)

```bash
wandb login  # if not already logged in
python src/train.py \
  training.epochs=5 \
  logging.use_wandb=true \
  experiment.name="validation-test" \
  experiment.tags=["validation"]
```

What to check in WandB dashboard:
- `train/loss_epoch` logged each epoch
- `train/grad_norm` logged
- `train/lr_group0` logged
- Config hyperparameters visible in run config

### 3g. Optimizer Parameter Groups

Verify per-component learning rates work:
```bash
python -c "
from src.core.torch_iekf import TorchIEKF
from src.models.init_process_cov_net import InitProcessCovNet
from src.models.measurement_cov_net import MeasurementCovNet
from src.training.optimizer_factory import build_optimizer

model = TorchIEKF()
model.initprocesscov_net = InitProcessCovNet()
model.mes_net = MeasurementCovNet(input_channels=6, output_dim=2)

cfg = {
    'type': 'Adam',
    'param_groups': {
        'init_process_cov_net': {'lr': 1e-3, 'weight_decay': 1e-8},
        'measurement_cov_net': {
            'cov_net': {'lr': 5e-4, 'weight_decay': 1e-8},
            'cov_lin': {'lr': 1e-4, 'weight_decay': 1e-8},
        }
    }
}
opt = build_optimizer(cfg, model)
for i, pg in enumerate(opt.param_groups):
    print(f'Group {i}: lr={pg[\"lr\"]}, weight_decay={pg[\"weight_decay\"]}, params={sum(p.numel() for p in pg[\"params\"])}')
"
```

Expected: 3 parameter groups with distinct learning rates (1e-3, 5e-4, 1e-4).

### 3h. Checkpoint Save/Load Roundtrip

```bash
python -c "
import torch, tempfile, os
from src.core.torch_iekf import TorchIEKF
from src.models.init_process_cov_net import InitProcessCovNet
from src.training.callbacks import CheckpointCallback

model = TorchIEKF()
model.initprocesscov_net = InitProcessCovNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

tmpdir = tempfile.mkdtemp()

class FakeTrainer:
    pass
trainer = FakeTrainer()
trainer.model = model
trainer.optimizer = opt

cb = CheckpointCallback(save_dir=tmpdir, save_interval=1, save_best=True, monitor='train/loss_epoch')
cb.on_train_start(trainer)
cb.on_epoch_end(trainer, 1, {'train/loss_epoch': 1.0})
cb.on_epoch_end(trainer, 2, {'train/loss_epoch': 0.5})

print(f'Checkpoints: {os.listdir(tmpdir)}')

# Reload
model2 = TorchIEKF()
model2.initprocesscov_net = InitProcessCovNet()
epoch, metrics = CheckpointCallback.load_checkpoint(os.path.join(tmpdir, 'best.pth'), model2)
print(f'Loaded best from epoch {epoch}, metrics: {metrics}')
"
```

---

## 4. Evaluation Smoke Tests

### 4a. Metrics on Synthetic Data

```python
python -c "
import numpy as np
from src.evaluation.metrics import compute_rpe, compute_ate, compute_orientation_error

N = 10000
dt = 0.01
speed = 10.0

# Ground truth: straight line
p_gt = np.zeros((N, 3))
p_gt[:, 0] = np.arange(N) * speed * dt
Rot_gt = np.tile(np.eye(3), (N, 1, 1))

# Prediction: ground truth + noise
p_pred = p_gt + np.random.randn(N, 3) * 0.5
Rot_pred = Rot_gt.copy()

rpe = compute_rpe(Rot_pred, p_pred, Rot_gt, p_gt)
ate = compute_ate(p_pred, p_gt, align=True)
orient = compute_orientation_error(Rot_pred, Rot_gt)

print('RPE:')
for k, v in rpe.items():
    print(f'  {k}: {v:.4f}')

print('\nATE:')
for k, v in ate.items():
    print(f'  {k}: {v:.4f}')

print('\nOrientation Error:')
for k, v in orient.items():
    print(f'  {k}: {v:.4f}')
"
```

What to check:
- RPE mean should be small (< 5% for 0.5m noise on a 1000m trajectory)
- ATE values should be in the range of the noise magnitude
- Orientation error should be ~0 degrees (no rotation error added)

### 4b. Visualization Output

Generate plots and verify they save correctly:
```python
python -c "
import numpy as np
import os
from src.evaluation.visualization import (
    plot_trajectory_2d, plot_trajectory_3d,
    plot_error_timeline, plot_covariance_timeline,
)

os.makedirs('results/validation', exist_ok=True)

N = 1000
t = np.linspace(0, 100, N)
p_gt = np.column_stack([100*np.cos(t*0.1), 100*np.sin(t*0.1), np.zeros(N)])
p_pred = p_gt + np.random.randn(N, 3) * 2

fig = plot_trajectory_2d(p_pred, p_gt, seq_name='validation')
fig.savefig('results/validation/traj_2d.png', dpi=150)
print('Saved results/validation/traj_2d.png')

fig = plot_trajectory_3d(p_pred, p_gt, seq_name='validation')
fig.savefig('results/validation/traj_3d.png', dpi=150)
print('Saved results/validation/traj_3d.png')

errors = np.linalg.norm(p_gt - p_pred, axis=1)
fig = plot_error_timeline(errors, timestamps=t, seq_name='validation')
fig.savefig('results/validation/errors.png', dpi=150)
print('Saved results/validation/errors.png')

covs = np.abs(np.random.randn(N, 2)) * 0.05
fig = plot_covariance_timeline(covs, timestamps=t, seq_name='validation')
fig.savefig('results/validation/covs.png', dpi=150)
print('Saved results/validation/covs.png')
"
```

Open the PNGs and visually inspect: trajectory should show circular path, errors should be noisy but bounded.

### 4c. Full Test Pipeline (Requires KITTI Data + Trained Checkpoint)

```bash
python src/test.py \
  checkpoint=checkpoints/best.pth \
  logging.use_wandb=false
```

What to check:
- RPE, ATE, orientation error printed per sequence
- Summary statistics printed at end
- Result pickles saved to `results/`
- Trajectory and error PNGs saved to `results/`

### 4d. Baseline Comparison (Requires Legacy Checkpoint)

Convert a legacy checkpoint and compare results:
```bash
# Convert
python scripts/convert_legacy_checkpoint.py \
  --input temp/iekfnets.p \
  --output checkpoints/converted.pth

# Test with converted checkpoint
python src/test.py \
  checkpoint=checkpoints/converted.pth \
  logging.use_wandb=false
```

Compare the RPE mean against the published result of **1.10%** translational error. The converted checkpoint should produce similar results (within ~5% tolerance).

---

## 5. Integration Tests

### 5a. End-to-End: Train Then Evaluate

Run a short training session and then evaluate the result:
```bash
# Train 5 epochs
python src/train.py \
  training.epochs=5 \
  logging.use_wandb=false \
  training.checkpointing.enabled=true

# Evaluate
python src/test.py \
  checkpoint=checkpoints/best.pth \
  logging.use_wandb=false
```

The RPE won't be good after 5 epochs, but it should be finite and the pipeline should complete without errors.

### 5b. Config Override Combinations

Verify various config overrides don't break anything:
```bash
# Different optimizer
python src/train.py training.epochs=1 training.debug.fast_dev_run=true \
  logging.use_wandb=false optimizer.type=AdamW

# Different loss
python src/train.py training.epochs=1 training.debug.fast_dev_run=true \
  logging.use_wandb=false loss.type=ATELoss

# Different learning rate
python src/train.py training.epochs=1 training.debug.fast_dev_run=true \
  logging.use_wandb=false optimizer.param_groups.init_process_cov_net.lr=1e-5
```

### 5c. RPE Loss Precomputation Consistency

Verify that precomputed RPE indices are correct by checking a round-trip:
```python
python -c "
import torch
from src.losses.rpe_loss import RPELoss
from src.core.torch_iekf import TorchIEKF

N = 5000
p = torch.zeros(N, 3).double()
p[:, 0] = torch.linspace(0, 500, N)
Rot = torch.eye(3).double().unsqueeze(0).expand(N, -1, -1).clone()

loss_fn = RPELoss()
list_rpe = loss_fn.precompute(Rot, p)

print(f'Index pairs: {len(list_rpe[0])}')
print(f'Delta p shape: {list_rpe[2].shape}')

# Perfect prediction should give zero loss
loss = loss_fn(Rot, p, Rot, p, list_rpe=list_rpe, N0=0)
print(f'Perfect prediction loss: {loss.item():.8f}')

# Noisy prediction should give positive loss
p_noisy = p + torch.randn_like(p) * 2.0
loss_noisy = loss_fn(Rot, p_noisy, Rot, p, list_rpe=list_rpe, N0=0)
print(f'Noisy prediction loss: {loss_noisy.item():.4f}')
"
```

---

## Quick Reference

| Test | Data Needed | Command |
|------|------------|---------|
| Unit tests (all) | None | `pytest tests/ -v` |
| Data loading | KITTI pickles | Section 2b |
| Training fast dev | None | `pytest tests/test_trainer.py -k fast_dev -v` |
| Training 1 epoch | KITTI pickles | Section 3c |
| Training with WandB | KITTI + WandB | Section 3f |
| Eval metrics | None | Section 4a |
| Eval visualization | None | Section 4b |
| Full test pipeline | KITTI + checkpoint | Section 4c |
| End-to-end | KITTI pickles | Section 5a |
