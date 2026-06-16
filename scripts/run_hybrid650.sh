#!/bin/bash
# Hybrid-650M RNA FM pretraining, RiNALMo-Giga-aligned, full RNAcentral, 8x L20.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=29651 \
  scripts/train_hybrid650_mlm.py \
  --output_dir outputs/fm_hybrid_650m \
  --micro_batch 8 --grad_accum 21 --epochs 6 \
  --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
  --max_len 1022 --num_workers 4
