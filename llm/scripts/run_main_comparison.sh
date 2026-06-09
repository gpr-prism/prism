#!/bin/bash
# =============================================================
# Main Comparison: GDN vs EFLA vs PGDN vs PRISM
#
# Runs all 4 models in parallel, each on NGPU_PER_MODEL GPUs.
# Automatically adjusts grad_accum_steps to keep effective
# batch size = 64 regardless of GPU count.
#
# Usage:
#   bash scripts/run_main_comparison.sh                  # default: 2 GPUs/model
#   NGPU_PER_MODEL=1 bash scripts/run_main_comparison.sh # 4-GPU machine
#   NGPU_PER_MODEL=4 bash scripts/run_main_comparison.sh # 16-GPU machine
#
# GPU layout (default NGPU_PER_MODEL=2, 8 GPUs total):
#   GDN   → GPU 0,1
#   EFLA  → GPU 2,3
#   PGDN  → GPU 4,5
#   PRISM → GPU 6,7
#
# Effective batch size = BATCH_SIZE * NGPU_PER_MODEL * GRAD_ACCUM = 64
# =============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# --------------- Configurable parameters ---------------
NGPU_PER_MODEL=${NGPU_PER_MODEL:-2}   # GPUs per model (default 2)
BATCH_SIZE=${BATCH_SIZE:-8}           # Per-GPU batch size
TARGET_EFFECTIVE_BATCH=64             # Keep this constant across GPU counts

# Auto-compute grad_accum_steps to maintain effective batch size
GRAD_ACCUM=$(( TARGET_EFFECTIVE_BATCH / (BATCH_SIZE * NGPU_PER_MODEL) ))
if [ "$GRAD_ACCUM" -lt 1 ]; then GRAD_ACCUM=1; fi
ACTUAL_EFFECTIVE=$(( BATCH_SIZE * NGPU_PER_MODEL * GRAD_ACCUM ))

# Base port (each model gets its own port: BASE_PORT, BASE_PORT+1, ...)
BASE_PORT=${BASE_PORT:-29500}
# -------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

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

# Build GPU assignment: 4 groups of NGPU_PER_MODEL GPUs
# e.g. NGPU_PER_MODEL=2 → "0,1"  "2,3"  "4,5"  "6,7"
#      NGPU_PER_MODEL=1 → "0"    "1"    "2"    "3"
build_gpu_list() {
    local start=$1
    local n=$2
    local list=""
    for ((i=0; i<n; i++)); do
        [ -n "$list" ] && list="${list},"
        list="${list}$((start + i))"
    done
    echo "$list"
}

GPU_GDN=$(build_gpu_list 0 $NGPU_PER_MODEL)
GPU_EFLA=$(build_gpu_list $((NGPU_PER_MODEL)) $NGPU_PER_MODEL)
GPU_PGDN=$(build_gpu_list $((NGPU_PER_MODEL * 2)) $NGPU_PER_MODEL)
GPU_PRISM=$(build_gpu_list $((NGPU_PER_MODEL * 3)) $NGPU_PER_MODEL)
TOTAL_GPUS=$((NGPU_PER_MODEL * 4))

echo "=============================================="
echo "  Main Comparison: GDN | EFLA | PGDN | PRISM"
echo "  NGPU_PER_MODEL = ${NGPU_PER_MODEL}"
echo "  Total GPUs     = ${TOTAL_GPUS}"
echo "  Per-GPU batch  = ${BATCH_SIZE}"
echo "  Grad accum     = ${GRAD_ACCUM}"
echo "  Effective batch= ${ACTUAL_EFFECTIVE}"
echo "  Start: $(date)"
echo "=============================================="

# GDN
echo "[$(date)] Starting GDN on GPU ${GPU_GDN}..."
CUDA_VISIBLE_DEVICES=${GPU_GDN} torchrun \
    --nproc_per_node=${NGPU_PER_MODEL} --master_port=${BASE_PORT} \
    train.py --models gdn ${COMMON_ARGS} \
    --ckpt_dir checkpoints/gdn \
    2>&1 | tee logs/gdn_${TIMESTAMP}.log &
PID_GDN=$!

# EFLA
echo "[$(date)] Starting EFLA on GPU ${GPU_EFLA}..."
CUDA_VISIBLE_DEVICES=${GPU_EFLA} torchrun \
    --nproc_per_node=${NGPU_PER_MODEL} --master_port=$((BASE_PORT + 1)) \
    train.py --models efla ${COMMON_ARGS} \
    --ckpt_dir checkpoints/efla \
    2>&1 | tee logs/efla_${TIMESTAMP}.log &
PID_EFLA=$!

# PGDN
echo "[$(date)] Starting PGDN on GPU ${GPU_PGDN}..."
CUDA_VISIBLE_DEVICES=${GPU_PGDN} torchrun \
    --nproc_per_node=${NGPU_PER_MODEL} --master_port=$((BASE_PORT + 2)) \
    train.py --models pgdn ${COMMON_ARGS} \
    --ckpt_dir checkpoints/pgdn \
    2>&1 | tee logs/pgdn_${TIMESTAMP}.log &
PID_PGDN=$!

# PRISM
echo "[$(date)] Starting PRISM on GPU ${GPU_PRISM}..."
CUDA_VISIBLE_DEVICES=${GPU_PRISM} torchrun \
    --nproc_per_node=${NGPU_PER_MODEL} --master_port=$((BASE_PORT + 3)) \
    train.py --models prism ${COMMON_ARGS} \
    --ckpt_dir checkpoints/prism \
    2>&1 | tee logs/prism_${TIMESTAMP}.log &
PID_PRISM=$!

echo ""
echo "All 4 models launched:"
echo "  GDN   PID=${PID_GDN}   (GPU ${GPU_GDN})"
echo "  EFLA  PID=${PID_EFLA}  (GPU ${GPU_EFLA})"
echo "  PGDN  PID=${PID_PGDN}  (GPU ${GPU_PGDN})"
echo "  PRISM PID=${PID_PRISM} (GPU ${GPU_PRISM})"
echo ""
echo "Waiting for all to finish..."

wait $PID_GDN;   STATUS_GDN=$?
wait $PID_EFLA;  STATUS_EFLA=$?
wait $PID_PGDN;  STATUS_PGDN=$?
wait $PID_PRISM; STATUS_PRISM=$?

echo ""
echo "=============================================="
echo "  ALL DONE at $(date)"
echo "  GDN:   exit=${STATUS_GDN}"
echo "  EFLA:  exit=${STATUS_EFLA}"
echo "  PGDN:  exit=${STATUS_PGDN}"
echo "  PRISM: exit=${STATUS_PRISM}"
echo "  Logs:  logs/"
echo "=============================================="
