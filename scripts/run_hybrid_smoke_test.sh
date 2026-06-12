#!/bin/bash
# Smoke test for the Transformer+Mamba hybrid backbone (does NOT start the real
# 1-epoch run). Steps: (1) count params vs BERT, (2) build + train 20 steps on a
# small streaming subset over 2 GPUs, (3) eval once, (4) confirm loss drops and
# artifacts are written. Uses GPUs 4,5 so it won't disturb other GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=/tmp/rna_fm_data
OUT=outputs/fm_hybrid_mamba_smoke
rm -rf "$OUT"

echo "=================================================================="
echo "[smoke] 1/2  param count (hybrid vs BERT baseline)"
echo "=================================================================="
python scripts/count_model_params.py --layer_pattern TTMTTM --mamba_expand 4

echo "=================================================================="
echo "[smoke] 2/2  train 20 steps + eval (2 GPUs)"
echo "=================================================================="
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29612 \
    scripts/train_hybrid_mamba_mlm.py \
    --tokenizer_type single --max_seq_length 512 \
    --train_file $DATA/train_single.txt --validation_file $DATA/val_single.txt \
    --vocab_dir tokenizers/single --output_dir "$OUT" \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --learning_rate 2e-4 --warmup_steps 2 --max_steps 20 \
    --eval_steps 10 --save_steps 20 --logging_steps 2 \
    --mlm_probability 0.15 --num_workers 2 --max_eval_samples 200 --streaming \
    --fp16 --layer_pattern TTMTTM --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 4

echo "=================================================================="
echo "[smoke] artifacts in $OUT:"
ls "$OUT" 2>/dev/null
echo "[smoke] eval_results.json:"
cat "$OUT/eval_results.json" 2>/dev/null
echo
echo "[smoke] DONE"
