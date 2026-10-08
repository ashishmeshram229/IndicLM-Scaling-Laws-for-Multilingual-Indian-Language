#!/bin/bash
# Single training job for WSAI cluster (V100-32GB, CUDA 11.8).
# Receives EXP_ID, MODEL_SIZE, TOKEN_BUDGET, SEED via --export=ALL from
# scaling_sweep.sh. Generates a YAML config at runtime and delegates to
# `indiclm experiment run`.
#
# MODEL_SIZE:   1m | 10m | 50m | 125m  (N_non_emb ≈ 1M / 10M / 50M / 125M)
# TOKEN_BUDGET: 50m | 200m | 500m | 1b
# SEED:         integer (default 0)
#
# Resume: on requeue the script automatically detects the latest step
# checkpoint under experiments/manifests/${EXP_ID}/checkpoints/ and passes
# it as resume_from, so wall-time kills are handled transparently.

#SBATCH --job-name=indiclm-train
#SBATCH --partition=public
#SBATCH --qos=gpu1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
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

# ── Model size → architecture ─────────────────────────────────────────────
# Targets N_non_emb ≈ label value (1M / 10M / 50M / 125M non-embedding params).
# Verified with ModelConfig.num_parameters_estimate() at vocab_size=32000.
case "${MODEL_SIZE}" in
    1m)   D_MODEL=128; N_LAYERS=5;  N_HEADS=4;  N_KV=2  ;;
    10m)  D_MODEL=256; N_LAYERS=14; N_HEADS=8;  N_KV=4  ;;
    50m)  D_MODEL=512; N_LAYERS=17; N_HEADS=8;  N_KV=4  ;;
    125m) D_MODEL=768; N_LAYERS=19; N_HEADS=12; N_KV=6  ;;
    *) echo "ERROR: Unknown MODEL_SIZE '${MODEL_SIZE}'. Use: 1m 10m 50m 125m" >&2; exit 1 ;;
esac

# ── Token budget → training steps ────────────────────────────────────────
# tokens_per_step = micro_batch(32) × grad_accum(4) × seq_len(512) = 65536
case "${TOKEN_BUDGET}" in
    50m)  MAX_STEPS=763;   TOTAL_TOKENS=50000000    ;;
    200m) MAX_STEPS=3052;  TOTAL_TOKENS=200000000   ;;
    500m) MAX_STEPS=7629;  TOTAL_TOKENS=500000000   ;;
    1b)   MAX_STEPS=15259; TOTAL_TOKENS=1000000000  ;;
    *) echo "ERROR: Unknown TOKEN_BUDGET '${TOKEN_BUDGET}'. Use: 50m 200m 500m 1b" >&2; exit 1 ;;
esac

WARMUP=$(( MAX_STEPS / 10 ))
EVAL_EVERY=$(( MAX_STEPS / 10 ))
LOG_EVERY=$(( MAX_STEPS / 50 < 5 ? 5 : MAX_STEPS / 50 ))
CKPT_EVERY=$(( MAX_STEPS / 4 < 100 ? 100 : MAX_STEPS / 4 ))

# ── Stage data to node-local SSD ──────────────────────────────────────────
STAGE_DIR="/tmp/indiclm_${SLURM_JOB_ID}"
mkdir -p "${STAGE_DIR}"
trap 'rm -rf "${STAGE_DIR}"' EXIT

echo "Staging processed data to ${STAGE_DIR}..."
rsync -a "${PROJECT}/data/processed/" "${STAGE_DIR}/processed/"

# ── Find latest checkpoint for resume (empty on first run) ────────────────
CKPT_DIR="${PROJECT}/experiments/manifests/${EXP_ID}/checkpoints"
LATEST_CKPT=$(ls -t "${CKPT_DIR}/step_"*.pt 2>/dev/null | head -1 || true)

# ── Generate per-run YAML config ──────────────────────────────────────────
CONFIG_FILE="${STAGE_DIR}/config.yaml"
cat > "${CONFIG_FILE}" << YAML_EOF
experiment_id: ${EXP_ID}
hypothesis: "Scaling law grid point: N=${MODEL_SIZE}, D=${TOKEN_BUDGET}, seed=${SEED}"
seed: ${SEED}
data:
  shards_dir: ${STAGE_DIR}/processed
  tokenizer_path: ${PROJECT}/data/tokenizer_v2/indiclm_tokenizer.model
  seq_len: 512
  total_tokens: ${TOTAL_TOKENS}
  alpha: 0.7
  dataset_version: v2
  tokenizer_version: bpe_v2
  eval_dir: ${PROJECT}/data/eval
model:
  d_model: ${D_MODEL}
  n_layers: ${N_LAYERS}
  n_heads: ${N_HEADS}
  n_kv_heads: ${N_KV}
  dropout: 0.0
training:
  max_steps: ${MAX_STEPS}
  micro_batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  warmup_steps: ${WARMUP}
  eval_every: ${EVAL_EVERY}
  checkpoint_every: ${CKPT_EVERY}
  log_every: ${LOG_EVERY}
  device: cuda
  precision: fp16
  wandb_project: indiclm-scaling
  wandb_run_name: ${EXP_ID}_${MODEL_SIZE}_${TOKEN_BUDGET}_s${SEED}
YAML_EOF

if [ -n "${LATEST_CKPT}" ]; then
    echo "  resume_from: ${LATEST_CKPT}" >> "${CONFIG_FILE}"
    echo "Resuming from checkpoint: ${LATEST_CKPT}"
fi

echo "Architecture: d_model=${D_MODEL}, n_layers=${N_LAYERS}, n_heads=${N_HEADS}, n_kv_heads=${N_KV}"
echo "Training:     max_steps=${MAX_STEPS}, warmup=${WARMUP}, checkpoint_every=${CKPT_EVERY}"
echo ""

# ── Run experiment ────────────────────────────────────────────────────────
python -m indiclm.cli.main experiment run \
    --config "${CONFIG_FILE}" \
    --seeds "${SEED}" \
    --experiments-root "${PROJECT}/experiments/manifests"

echo "Training complete for ${EXP_ID} (${MODEL_SIZE}, ${TOKEN_BUDGET}, seed=${SEED})"
