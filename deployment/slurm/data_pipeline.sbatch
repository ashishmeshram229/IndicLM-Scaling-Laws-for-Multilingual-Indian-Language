#!/bin/bash
# CPU data pipeline job: Wikipedia parquet → quality filter → dedup → shards.
# Runs on the CPU-only node (no --gres). Submit before training jobs.
#
# Usage:
#   sbatch deployment/slurm/data_pipeline.sbatch
#
# Output: /storage_server/da25m016/indiclm/data/processed/

#SBATCH --job-name=indiclm-data
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=0-12:00:00
#SBATCH --output=/storage_server/da25m016/indiclm/logs/data-%x-%j.out
#SBATCH --error=/storage_server/da25m016/indiclm/logs/data-%x-%j.err

set -eo pipefail

PROJECT=/storage_server/da25m016/indiclm
source "${PROJECT}/env/bin/activate"

# Compute nodes have no internet — all data must already be in HF cache
export HF_HOME="${PROJECT}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

RAW_WIKI="${PROJECT}/data/raw_wiki"
OUTPUT_DIR="${PROJECT}/data/processed"
TOKENIZER_DIR="${PROJECT}/data/tokenizer_v2"
VERSION="${DATASET_VERSION:-v2}"

echo "=========================================="
echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "RAM:    ${SLURM_MEM_PER_NODE:-?} MB"
echo "Version: ${VERSION}"
echo "=========================================="

# Verify raw wiki parquet files exist
echo "Checking raw wiki parquet files..."
for lang in hi mr bn ta te kn ml gu pa en; do
    parquet="${RAW_WIKI}/${lang}/wiki.parquet"
    if [ ! -f "${parquet}" ]; then
        echo "ERROR: Missing ${parquet}. Run scripts/download_corpus.py first."
        exit 1
    fi
    size=$(du -sh "${parquet}" | cut -f1)
    echo "  [${lang}] ${size}"
done
echo "All parquet files present."

# Skip if output already exists
if [ -f "${OUTPUT_DIR}/pipeline_stats.json" ]; then
    echo "Pipeline output already exists at ${OUTPUT_DIR} — skipping data pipeline."
else
    echo ""
    echo "Running Wikipedia data pipeline..."
    echo "Processing languages: hi mr bn ta te kn ml gu pa en"
    echo "English capped at 500K articles (set --max-docs-per-language=0 to disable)"
    echo ""

    indiclm data prepare \
        --source-format wikipedia-parquet \
        --raw-wiki-dir "${RAW_WIKI}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset-version "${VERSION}" \
        --languages "hi,mr,bn,ta,te,kn,ml,gu,pa,en" \
        --enable-exact-dedup true \
        --enable-near-dedup true \
        --enable-quality-filter true \
        --min-quality-score 0.5

    echo ""
    echo "Data pipeline complete. Printing stats..."
    indiclm data stats --output-dir "${OUTPUT_DIR}"
fi

# Train tokenizer on the fresh processed corpus
if [ -d "${TOKENIZER_DIR}" ]; then
    echo "Tokenizer already exists at ${TOKENIZER_DIR} — skipping."
else
    echo ""
    echo "Training 32K BPE tokenizer on processed corpus..."
    indiclm tokenizer train \
        --data-dir "${OUTPUT_DIR}" \
        --output-dir "${TOKENIZER_DIR}" \
        --vocab-size 32000 \
        --model-type bpe
    echo "Tokenizer complete: ${TOKENIZER_DIR}"
fi

echo ""
echo "Data pipeline job complete."
echo "Next: bash deployment/slurm/scaling_sweep.sh"
