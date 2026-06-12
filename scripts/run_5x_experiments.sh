#!/bin/bash
# Sequentially run two 100k-step (5x) MLM experiments on the 1M-record dataset:
#   1) single  (1 base/token)
#   2) kmer3   (non-overlapping 3-mer, leak-free)
# then render the comparison chart. Designed to be launched detached
# (setsid + nohup) so it survives the controlling session disconnecting.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

COMMON="--max_seq_length 512 \
  --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
  --learning_rate 1e-4 --weight_decay 0.01 --warmup_steps 1000 \
  --max_steps 100000 --eval_steps 250 --save_steps 10000 --logging_steps 50 \
  --mlm_probability 0.15 --num_workers 8 --max_eval_samples 2000"

echo "[5x] ===== single (100k) START $(date -Is) ====="
torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
    --tokenizer_type single \
    --train_file data/processed/mlm/train_single.txt \
    --validation_file data/processed/mlm/val_single.txt \
    --vocab_dir tokenizers/single \
    --output_dir outputs/bert_mlm/single_100k \
    $COMMON
echo "[5x] ===== single (100k) DONE  $(date -Is) ====="

echo "[5x] ===== kmer3 (100k) START $(date -Is) ====="
torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
    --tokenizer_type kmer3 \
    --train_file data/processed/mlm/train_kmer3.txt \
    --validation_file data/processed/mlm/val_kmer3.txt \
    --vocab_dir tokenizers/kmer3 \
    --output_dir outputs/bert_mlm/kmer3_100k \
    $COMMON
echo "[5x] ===== kmer3 (100k) DONE  $(date -Is) ====="

python scripts/plot_compare.py || true
echo "[5x] ===== ALL DONE $(date -Is) ====="
