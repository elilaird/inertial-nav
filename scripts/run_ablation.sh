#!/bin/bash
# Posterior validation ablation study: train 4 conditions and evaluate.
#
# Condition A: AI-IMU baseline (MeasurementCovNet + InitProcessCovNet)
# Condition B: World model, measurement head only
# Condition C: World model, process head only (fixed N_n)
# Condition D: World model, both heads
#
# Usage:
#   bash scripts/run_ablation.sh          # Train all 4 conditions
#   bash scripts/run_ablation.sh --test   # Test only (skip training)

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=400
COMMON_ARGS="training.epochs=${EPOCHS} logging.use_wandb=true"

TEST_ONLY=false
if [[ "${1:-}" == "--test" ]]; then
    TEST_ONLY=true
fi

# ---- Training ----
if [[ "$TEST_ONLY" == false ]]; then
    echo "=== Training Condition A: AI-IMU baseline ==="
    python src/train.py model=iekf_learned_cov \
        experiment.name=ablation_A experiment.group=posterior_validation \
        $COMMON_ARGS

    echo "=== Training Condition B: WM measurement head only ==="
    python src/train.py model=iekf_wm_meas_only \
        experiment.name=ablation_B experiment.group=posterior_validation \
        $COMMON_ARGS

    echo "=== Training Condition C: WM process head only ==="
    python src/train.py model=iekf_wm_proc_only \
        experiment.name=ablation_C experiment.group=posterior_validation \
        $COMMON_ARGS

    echo "=== Training Condition D: WM both heads ==="
    python src/train.py model=iekf_world_model \
        experiment.name=ablation_D experiment.group=posterior_validation \
        $COMMON_ARGS
fi

# ---- Testing ----
# After training, find the best checkpoint for each condition and evaluate
# on the 7 ablation test sequences.
#
# NOTE: You must set CKPT_A/B/C/D to the actual checkpoint paths from training.
# Hydra outputs go to outputs/<date>/<time>/checkpoints/best.pth
echo ""
echo "=== Testing ==="
echo "Set checkpoint paths before running test phase:"
echo "  export CKPT_A=outputs/<date>/<time>/checkpoints/best.pth"
echo "  export CKPT_B=outputs/<date>/<time>/checkpoints/best.pth"
echo "  export CKPT_C=outputs/<date>/<time>/checkpoints/best.pth"
echo "  export CKPT_D=outputs/<date>/<time>/checkpoints/best.pth"

if [[ -n "${CKPT_A:-}" ]]; then
    echo "=== Testing Condition A ==="
    python src/test.py model=iekf_learned_cov dataset=kitti_ablation_test \
        checkpoint="$CKPT_A" experiment.name=ablation_A_test \
        logging.use_wandb=true
fi

if [[ -n "${CKPT_B:-}" ]]; then
    echo "=== Testing Condition B ==="
    python src/test.py model=iekf_wm_meas_only dataset=kitti_ablation_test \
        checkpoint="$CKPT_B" experiment.name=ablation_B_test \
        logging.use_wandb=true
fi

if [[ -n "${CKPT_C:-}" ]]; then
    echo "=== Testing Condition C ==="
    python src/test.py model=iekf_wm_proc_only dataset=kitti_ablation_test \
        checkpoint="$CKPT_C" experiment.name=ablation_C_test \
        logging.use_wandb=true
fi

if [[ -n "${CKPT_D:-}" ]]; then
    echo "=== Testing Condition D ==="
    python src/test.py model=iekf_world_model dataset=kitti_ablation_test \
        checkpoint="$CKPT_D" experiment.name=ablation_D_test \
        logging.use_wandb=true
fi
