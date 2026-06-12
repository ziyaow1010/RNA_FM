#!/bin/bash
# 8x L20 BERT MLM pretraining with the NON-OVERLAPPING 3-mer tokenizer (stride 3).
# Adjacent tokens share no bases, so plain single-token MLM masking has no
# neighbor leakage -> a fair, leak-free comparison against the single tokenizer.
# If you hit OOM, change both batch-size values from 64 to 32.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
    --tokenizer_type kmer3 \
    --train_file data/processed/mlm/train_kmer3.txt \
    --validation_file data/processed/mlm/val_kmer3.txt \
    --vocab_dir tokenizers/kmer3 \
    --output_dir outputs/bert_mlm/kmer3 \
    --max_seq_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --max_steps 20000 \
    --eval_steps 250 \
    --save_steps 5000 \
    --logging_steps 50 \
    --mlm_probability 0.15 \
    --num_workers 8 \
    --max_eval_samples 2000
