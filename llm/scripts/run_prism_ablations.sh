#!/bin/bash
# =============================================================
# PRISM Ablation Study
# Tests 5 variants serially, each using NGPU GPUs.
# Automatically adjusts grad_accum_steps to keep effective
# batch size = 64 regardless of GPU count.
#
# Usage:
#   bash scripts/run_prism_ablations.sh            # default: 8 GPUs
#   NGPU=4 bash scripts/run_prism_ablations.sh     # 4-GPU machine
#   NGPU=1 bash scripts/run_prism_ablations.sh     # single GPU
#
# Variants:
#   prism    — full PRISM (independent K per step, closed-form retain)
#   prism_r2 — shared K across all solver steps
#   prism_r3 — no retain (no closed-form cumulative product)
#   prism_r4 — step 0 reuses K_base from main path
#   prism_r5 — ALL steps reuse K_base (no independent K projections)
# =============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# --------------- Configurable parameters ---------------
NGPU=${NGPU:-8}                       # Total GPUs to use (default 8)
BATCH_SIZE=${BATCH_SIZE:-8}           # Per-GPU batch size
TARGET_EFFECTIVE_BATCH=64             # Keep this constant across GPU counts

# Auto-compute grad_accum_steps to maintain effective batch size
GRAD_ACCUM=$(( TARGET_EFFECTIVE_BATCH / (BATCH_SIZE * NGPU) ))
if [ "$GRAD_ACCUM" -lt 1 ]; then GRAD_ACCUM=1; fi
ACTUAL_EFFECTIVE=$(( BATCH_SIZE * NGPU * GRAD_ACCUM ))
# -------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ablation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Auto-detect data directory
DATA_FLAG=""
if [ -d "data/train" ]; then
    DATA_FLAG="--data_dir data"
fi

COMMON_ARGS="--config large_130m --epochs 10 \
    --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} \
    --lr 6e-5 --weight_decay 0.05 \
    --warmup_steps 1500 --grad_clip 0.5 \
    ${DATA_FLAG}"

MODELS=(prism prism_r2 prism_r3 prism_r4 prism_r5)
DESCRIPTIONS=(
    "PRISM full (independent K, closed-form retain)"
    "PRISM-r2: shared K across all solver steps"
    "PRISM-r3: no retain (no cumulative product)"
    "PRISM-r4: step 0 reuses K_base"
    "PRISM-r5: ALL steps reuse K_base (no step_k_proj)"
)

NUM_TASKS=${#MODELS[@]}

echo "=============================================="
echo "  PRISM Ablation Study"
echo "  NGPU           = ${NGPU}"
echo "  Per-GPU batch  = ${BATCH_SIZE}"
echo "  Grad accum     = ${GRAD_ACCUM}"
echo "  Effective batch= ${ACTUAL_EFFECTIVE}"
echo "  $NUM_TASKS variants, serial"
echo "  Start: $(date)"
echo "=============================================="

for i in $(seq 0 $((NUM_TASKS - 1))); do
    MODEL=${MODELS[$i]}
    DESC=${DESCRIPTIONS[$i]}

    echo ""
    echo "┌──────────────────────────────────────────────┐"
    echo "│  Task $((i+1))/$NUM_TASKS: $MODEL"
    echo "│  $DESC"
    echo "│  Start: $(date)"
    echo "└──────────────────────────────────────────────┘"

    torchrun \
        --nproc_per_node=${NGPU} --master_port=29500 \
        train.py --models "$MODEL" \
        ${COMMON_ARGS} \
        --ckpt_dir "checkpoints/ablation_${MODEL}" \
        2>&1 | tee "$LOG_DIR/${MODEL}.log"

    STATUS=$?
    echo "[Task $((i+1))/$NUM_TASKS] $MODEL finished with exit=$STATUS at $(date)"
done

echo ""
echo "=============================================="
echo "  ALL $NUM_TASKS ABLATIONS COMPLETE"
echo "  End: $(date)"
echo "  Logs: $LOG_DIR/"
echo "=============================================="
