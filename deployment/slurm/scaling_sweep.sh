#!/bin/bash
# Scaling sweep orchestrator.
#
# Chains: data pipeline → tokenizer → training grid (all sizes × budgets × seeds)
# using SLURM job dependencies. Prints a summary of submitted job IDs.
#
# Usage (run on the login node):
#   bash deployment/slurm/scaling_sweep.sh
#
# Override defaults via env vars:
#   SEEDS="0 1 2"
#   MODEL_SIZES="1m 10m 50m 125m"
#   TOKEN_BUDGETS="50m 200m 500m 1b"
#   START_EXP=13

set -eo pipefail

PROJECT=/storage_server/da25m016/indiclm
LOGS="${PROJECT}/logs"
mkdir -p "${LOGS}"

# Sweep configuration
SEEDS="${SEEDS:-0 1 2}"
MODEL_SIZES="${MODEL_SIZES:-1m 10m 50m 125m}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-50m 200m 500m 1b}"
START_EXP="${START_EXP:-13}"

echo "=========================================="
echo "IndicLM Scaling Sweep"
echo "Model sizes:   ${MODEL_SIZES}"
echo "Token budgets: ${TOKEN_BUDGETS}"
echo "Seeds:         ${SEEDS}"
echo "Starting EXP:  EXP-$(printf '%03d' ${START_EXP})"
echo "=========================================="

# Step 1: Submit data pipeline job (or reuse an existing completed one)
echo ""
if [ -n "${DATA_JID:-}" ]; then
    echo "  data_pipeline → reusing job ${DATA_JID} (skipping resubmission)"
else
    echo "Submitting data pipeline job..."
    DATA_JID=$(sbatch --parsable "${PROJECT}/deployment/slurm/data_pipeline.sbatch")
    echo "  data_pipeline → job ${DATA_JID}"
fi

# Step 2: Submit training jobs, each depending on data pipeline completing
exp_num=${START_EXP}
declare -a TRAIN_JIDS

for size in ${MODEL_SIZES}; do
    for budget in ${TOKEN_BUDGETS}; do
        for seed in ${SEEDS}; do
            EXP_ID="EXP-$(printf '%03d' ${exp_num})"

            # Pass multi-value args via env vars (sbatch --export splits on commas)
            JID=$(
                EXP_ID="${EXP_ID}" \
                MODEL_SIZE="${size}" \
                TOKEN_BUDGET="${budget}" \
                SEED="${seed}" \
                sbatch \
                    --parsable \
                    --export=ALL \
                    --job-name="indiclm-${EXP_ID}-${size}-${budget}-s${seed}" \
                    --dependency="afterok:${DATA_JID}" \
                    "${PROJECT}/deployment/slurm/train_gpu.sbatch"
            )

            TRAIN_JIDS+=("${JID}")
            echo "  ${EXP_ID} (${size}, ${budget}, seed=${seed}) → job ${JID}"
            exp_num=$((exp_num + 1))
        done
    done
done

# Step 3: Submit aggregation job once all training jobs complete
ALL_TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JIDS[*]}")
END_EXP=$(( START_EXP + ${#TRAIN_JIDS[@]} - 1 ))
echo ""
echo "Submitting aggregation job (depends on all training jobs)..."

AGG_JID=$(sbatch --parsable \
    --job-name=indiclm-aggregate \
    --partition=public \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --dependency="afterok:${ALL_TRAIN_DEPS}" \
    --output="${LOGS}/aggregate-%j.out" \
    --error="${LOGS}/aggregate-%j.err" \
    --wrap="
        set -eo pipefail
        source ${PROJECT}/env/bin/activate
        cd ${PROJECT}
        mkdir -p ${PROJECT}/experiments/scaling_results
        python -m indiclm.cli.main experiment aggregate-scaling \
            --manifests-dir ${PROJECT}/experiments/manifests \
            --output-dir ${PROJECT}/experiments/scaling_results \
            --start-exp ${START_EXP} \
            --end-exp ${END_EXP}
        echo 'Aggregation complete. Results: ${PROJECT}/experiments/scaling_results/'
    "
)

echo "  aggregation → job ${AGG_JID}"
echo ""
echo "=========================================="
echo "Total training jobs: ${#TRAIN_JIDS[@]}"
echo "Monitor with: squeue -u \$USER"
echo "Tail logs:    tail -f ${LOGS}/train-indiclm-train-<JID>.out"
echo "=========================================="
