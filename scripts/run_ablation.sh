#!/bin/bash
# Ablation study comparing three experimental conditions:
# 1. baseline      (classical IEKF, no learned components)
# 2. learned_cov   (InitProcessCovNet + MeasurementCovNet, no bias correction)
# 3. learned_dynamics  (learned covariances + LearnedBiasCorrectionNet)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
EPOCHS=400
SEQUENCES="00,01,04,06,07,08,09,10"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/ablation"
RESULTS_DIR="$PROJECT_ROOT/results/ablation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "====================================================================="
echo "Ablation Study: Learned Bias Correction Network"
echo "====================================================================="
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Test Sequences: $SEQUENCES"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  Results Dir: $RESULTS_DIR"
echo ""

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

# Helper function to run train/test for a condition
run_condition() {
    local CONDITION=$1
    local MODEL_CONFIG=$2
    local DESC=$3

    echo "====================================================================="
    echo "[$CONDITION] $DESC"
    echo "====================================================================="

    CHECKPOINT_PATH="$CHECKPOINT_DIR/${CONDITION}_best.pth"
    CONDITION_RESULTS="$RESULTS_DIR/$CONDITION"
    mkdir -p "$CONDITION_RESULTS"

    # Training phase
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Training $CONDITION model..."
        cd "$PROJECT_ROOT"
        python src/train.py \
            model="$MODEL_CONFIG" \
            training.epochs="$EPOCHS" \
            training.checkpoint_every=20 \
            training.checkpoint_dir="$CHECKPOINT_DIR" \
            wandb.mode=offline \
            wandb.name="ablation_${CONDITION}" \
            2>&1 | tee "$CONDITION_RESULTS/train.log"

        echo "Training complete for $CONDITION"
    else
        echo "Using existing checkpoint: $CHECKPOINT_PATH"
    fi

    # Testing phase
    echo "Testing $CONDITION model on sequences: $SEQUENCES"
    cd "$PROJECT_ROOT"
    python src/test.py \
        model="$MODEL_CONFIG" \
        checkpoint="$CHECKPOINT_PATH" \
        output_dir="$CONDITION_RESULTS" \
        sequences="$SEQUENCES" \
        wandb.mode=offline \
        2>&1 | tee "$CONDITION_RESULTS/test.log"

    echo "$CONDITION evaluation complete"
    echo ""
}

# Run all three conditions
run_condition "baseline" "iekf_baseline" "Classical IEKF (no learned components)"
run_condition "learned_cov" "iekf_learned_cov" "Learned covariances (no bias correction)"
run_condition "learned_dynamics" "iekf_learned_dynamics" "Full system (learned covariances + bias correction)"

echo "====================================================================="
echo "Generating Comparison Table"
echo "====================================================================="

cd "$PROJECT_ROOT"
python scripts/compare_ablation_results.py \
    --baseline "$RESULTS_DIR/baseline" \
    --learned_cov "$RESULTS_DIR/learned_cov" \
    --learned_dynamics "$RESULTS_DIR/learned_dynamics" \
    --sequences "$SEQUENCES" \
    --output "$RESULTS_DIR/comparison.txt"

echo ""
echo "====================================================================="
echo "Ablation Study Complete!"
echo "====================================================================="
echo "Results saved to: $RESULTS_DIR"
echo "Comparison table: $RESULTS_DIR/comparison.txt"
echo ""
