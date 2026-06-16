#!/bin/bash
# Build the rnafm conda environment on a CPU node.
# Strict version pins from HANDOFF §5.
#SBATCH --job-name=rnafm_env
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/build_env_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/build_env_%j.err

set -euo pipefail
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
ENV_DIR=$CONDA_BASE/envs/rnafm

echo "[env] Starting at $(date)"

# Use full path conda to avoid relying on PATH
CONDA_BIN=$CONDA_BASE/bin/conda

# Create env if not exists
if [ -d "$ENV_DIR" ]; then
    echo "[env] Env already exists at $ENV_DIR, skipping create."
else
    $CONDA_BIN create -y -n rnafm python=3.11
    echo "[env] Created rnafm env"
fi

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
PIP="$ENV_DIR/bin/pip"

echo "[env] Python: $(python --version)"
echo "[env] Installing torch 2.5.1+cu121..."
$PIP install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

echo "[env] Installing transformers 4.46.3, datasets 3.1.0..."
$PIP install \
    transformers==4.46.3 \
    datasets==3.1.0 \
    "accelerate>=0.34,<1.2" \
    "tokenizers>=0.20,<0.21" \
    --quiet

echo "[env] Installing other dependencies..."
$PIP install matplotlib tqdm pandas numpy scipy --quiet

echo "[env] Installing mmseqs2 via conda (for clustering step)..."
$CONDA_BIN install -y -n rnafm -c bioconda mmseqs2 --quiet 2>/dev/null || \
    echo "[env] mmseqs2 conda install failed (no internet?), will try binary fallback in cluster step"

echo "[env] Installing causal-conv1d 1.5.0.post8 (prebuilt wheel, --no-deps)..."
$PIP install --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "[env] Installing mamba-ssm 2.2.4 (prebuilt wheel, --no-deps)..."
$PIP install --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "[env] Installing flash-attn 2.7.4.post1 (prebuilt wheel, --no-deps)..."
$PIP install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "[env] Verifying imports..."
python -c "
import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
import transformers; print(f'transformers {transformers.__version__}')
import datasets; print(f'datasets {datasets.__version__}')
import mamba_ssm; print('mamba_ssm OK')
import causal_conv1d; print('causal_conv1d OK')
import flash_attn; print(f'flash_attn {flash_attn.__version__}')
"

echo "[env] Done at $(date)"
