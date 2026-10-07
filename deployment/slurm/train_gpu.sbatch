#!/bin/bash
# Single-GPU training job for WSAI cluster (V100-32GB, CUDA 11.8).
#
# Usage:
#   sbatch --export=ALL,EXP_ID=EXP-013,MODEL_SIZE=50m,TOKEN_BUDGET=200m,SEED=0 \
#          deployment/slurm/train_gpu.sbatch
#
# Required env vars (pass via --export=ALL,VAR=val):
#   EXP_ID        experiment identifier, e.g. EXP-013
#   MODEL_SIZE    one of: 1m 10m 50m 125m 350m
#   TOKEN_BUDGET  token budget, e.g. 50m 200m 500m 1b
#   SEED          random seed (default: 0)
#
# Output: /storage_server/da25m016/indiclm/experiments/manifests/$EXP_ID/

#SBATCH --job-name=indiclm-train
#SBATCH --partition=public
#SBATCH --qos=gpu1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=/storage_server/da25m016/indiclm/logs/train-%x-%j.out
#SBATCH --error=/storage_server/da25m016/indiclm/logs/train-%x-%j.err

set -eo pipefail

PROJECT=/storage_server/da25m016/indiclm
source "${PROJECT}/env/bin/activate"

export HF_HOME="${PROJECT}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Default seed if not set
SEED="${SEED:-0}"

echo "=========================================="
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "EXP_ID:   ${EXP_ID}"
echo "SIZE:     ${MODEL_SIZE}"
echo "BUDGET:   ${TOKEN_BUDGET}"
echo "SEED:     ${SEED}"
echo "=========================================="

# Auto-requeue ~15 min before wall time limit
trap 'echo "SIGTERM received — requeueing job"; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1

# Stage data to node-local SSD for fast I/O
STAGE_DIR="/tmp/indiclm_${SLURM_JOB_ID}"
mkdir -p "${STAGE_DIR}"
echo "Staging processed data to ${STAGE_DIR}..."
rsync -a --progress "${PROJECT}/data/processed/" "${STAGE_DIR}/processed/"

# Run training in background so USR1 trap fires
python -m indiclm.cli.main train \
    --exp-id "${EXP_ID}" \
    --model-size "${MODEL_SIZE}" \
    --token-budget "${TOKEN_BUDGET}" \
    --seed "${SEED}" \
    --device cuda \
    --precision fp16 \
    --data-dir "${STAGE_DIR}/processed" \
    --output-dir "${PROJECT}/experiments/manifests/${EXP_ID}/seed_${SEED}" \
    --wandb-project indiclm-scaling \
    --wandb-run-name "${EXP_ID}_${MODEL_SIZE}_${TOKEN_BUDGET}_s${SEED}" \
    --resume-if-checkpoint &

TRAIN_PID=$!
wait "${TRAIN_PID}"

# Cleanup stage dir
rm -rf "${STAGE_DIR}"
echo "Training complete for ${EXP_ID} (${MODEL_SIZE}, ${TOKEN_BUDGET}, seed=${SEED})"
