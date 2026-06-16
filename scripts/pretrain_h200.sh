#!/bin/bash
# Hybrid-650M pretraining on H200 (1 node × 8 GPU × 141 GB each).
# RiNALMo-aligned: single-nuc tokenizer, ctx 1022, MLM 15%, eff-batch 1344, 6 epochs.
# Clustering step (HANDOFF §3b) recommended first: run cluster_rnacentral.sh.
#
# Usage: sbatch scripts/pretrain_h200.sh
# Resumes automatically if outputs/fm_hybrid_650m/checkpoint-* exists.
#SBATCH --job-name=rnafm_pretrain
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h200_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h200_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

# Symlink checkpoint dir so train script writes there (avoids home dir space issue)
if [ ! -L outputs/fm_hybrid_650m ] && [ ! -d outputs/fm_hybrid_650m/.git ]; then
    rm -rf outputs/fm_hybrid_650m
    ln -s "$CKPT_DIR" outputs/fm_hybrid_650m
    echo "[pretrain] Symlinked outputs/fm_hybrid_650m -> $CKPT_DIR"
fi

# Symlink data dir
mkdir -p data/raw
if [ ! -e data/raw/rnacentral_active.fasta.gz ]; then
    # Use clustered data if available (~17M reps, matching RiNALMo)
    if [ -f "$DATA_DIR/clustered_17m.fasta" ]; then
        # rna_stream_dataset expects .fasta.gz; compress the clustered version
        echo "[pretrain] Compressing clustered fasta..."
        pigz -k -p 8 "$DATA_DIR/clustered_17m.fasta" || gzip -k "$DATA_DIR/clustered_17m.fasta"
        ln -sf "$DATA_DIR/clustered_17m.fasta.gz" data/raw/rnacentral_active.fasta.gz
        echo "[pretrain] Using clustered data (~17M reps)"
    elif [ -f "$DATA_DIR/rnacentral_active.fasta.gz" ]; then
        ln -sf "$DATA_DIR/rnacentral_active.fasta.gz" data/raw/rnacentral_active.fasta.gz
        echo "[pretrain] WARNING: using full 40.7M (unclustered) — run cluster_rnacentral.sh first for fair comparison"
    else
        echo "[pretrain] ERROR: no RNAcentral data found at $DATA_DIR" >&2
        exit 1
    fi
fi

echo "[pretrain] Node: $(hostname)"
echo "[pretrain] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain] Start: $(date)"

# Step 1: Throughput benchmark to pick optimal micro_batch
if [ ! -f "$CKPT_DIR/h200_bench.json" ]; then
    echo "[pretrain] Running throughput benchmark..."
    python scripts/bench_throughput.py
fi

# Read recommended micro_batch from bench results
MICRO_BATCH=$(python3 -c "
import json
r = json.load(open('$CKPT_DIR/h200_bench.json'))
print(r['best_micro_batch'])
" 2>/dev/null || echo 32)
GRAD_ACCUM=$(python3 -c "
import json
r = json.load(open('$CKPT_DIR/h200_bench.json'))
print(r['best_grad_accum'])
" 2>/dev/null || echo 5)

echo "[pretrain] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=8"
echo "[pretrain] Effective batch = $((MICRO_BATCH * 8 * GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=29651 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_650m \
    --gz data/raw/rnacentral_active.fasta.gz \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs 6 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 8

echo "[pretrain] Done: $(date)"
