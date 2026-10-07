#!/bin/bash
# One-shot cluster environment bootstrap.
# Run on the WSAI login node (10.24.6.55) as user da25m016.
#
# What it does:
#   1. Creates the project directory structure on /storage_server
#   2. Creates a venv inheriting torch 2.7.1+cu118 from the system Python
#      (never reinstalls PyTorch — it is pinned via constraints.txt)
#   3. Generates constraints.txt to lock torch/numpy/nvidia-* versions
#   4. Installs indiclm with all extras: gpu, monitoring, tracking
#   5. Verifies the install and GPU visibility
#
# Usage (from your local Mac):
#   rsync -av --partial --progress \
#       /Users/ashishmeshram/Desktop/Project-Placements/indiclm/ \
#       da25m016@10.24.6.55:/storage_server/da25m016/indiclm/
#   ssh da25m016@10.24.6.55 "bash /storage_server/da25m016/indiclm/scripts/setup_cluster.sh"

set -eo pipefail

BASE_PYTHON=/storage/opt/miniforge3/bin/python3
PROJECT=/storage_server/da25m016/indiclm
VENV="${PROJECT}/env"
LOGS="${PROJECT}/logs"

echo "============================================"
echo "IndicLM Cluster Bootstrap"
echo "Project: ${PROJECT}"
echo "Python:  $(${BASE_PYTHON} --version)"
echo "============================================"

# 1. Directory structure
echo ""
echo "[1/5] Creating directory structure..."
mkdir -p "${LOGS}"
mkdir -p "${PROJECT}/data/raw_wiki"
mkdir -p "${PROJECT}/data/processed"
mkdir -p "${PROJECT}/data/tokenizer_v2"
mkdir -p "${PROJECT}/experiments/manifests"
mkdir -p "${PROJECT}/experiments/reports"
mkdir -p "${PROJECT}/hf_cache"
echo "  Done."

# 2. Create venv inheriting system torch (--system-site-packages is crucial)
echo ""
echo "[2/5] Creating venv at ${VENV}..."
if [ -d "${VENV}" ]; then
    echo "  Venv already exists — skipping creation."
else
    "${BASE_PYTHON}" -m venv --system-site-packages "${VENV}"
    echo "  Venv created."
fi
source "${VENV}/bin/activate"

# 3. Generate constraints.txt to lock torch/torchvision/numpy/nvidia-*
echo ""
echo "[3/5] Generating constraints.txt..."
python - > "${PROJECT}/constraints.txt" <<'EOF'
import importlib.metadata as md
print("\n".join(
    f"{d.metadata['Name']}=={d.version}"
    for d in md.distributions()
    if d.metadata['Name'] and (
        d.metadata['Name'].lower() in {'torch','torchvision','torchaudio','numpy','triton'}
        or d.metadata['Name'].lower().startswith('nvidia-')
    )
))
EOF
echo "  constraints.txt written:"
cat "${PROJECT}/constraints.txt"

# 4. Install indiclm with all extras
echo ""
echo "[4/5] Installing indiclm[gpu,monitoring,tracking]..."
cd "${PROJECT}"
pip install --no-cache-dir -c constraints.txt -e ".[gpu,monitoring,tracking]"
echo "  Done."

# 5. Verify
echo ""
echo "[5/5] Verification..."
python -c "
import torch, indiclm
print(f'  indiclm version : {indiclm.__version__}')
print(f'  torch version   : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    print(f'  Compute cap.    : {torch.cuda.get_device_capability(0)}')
    print(f'  fp16 support    : {torch.cuda.get_device_capability(0)[0] >= 7}')
    print(f'  bf16 support    : {torch.cuda.get_device_capability(0)[0] >= 8}')
"

echo ""
echo "============================================"
echo "Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Download Wikipedia corpus (login node only, has internet):"
echo "     python scripts/download_corpus.py \\"
echo "         --hf-cache ${PROJECT}/hf_cache \\"
echo "         --output-dir ${PROJECT}/data/raw_wiki"
echo ""
echo "  2. Submit the full scaling sweep:"
echo "     bash deployment/slurm/scaling_sweep.sh"
echo ""
echo "  3. Monitor:"
echo "     squeue -u \$USER"
echo "     tail -f ${LOGS}/train-indiclm-train-<JID>.out"
echo "============================================"
