#!/bin/bash
# 8x L20 BERT MLM pretraining with the single-base tokenizer.
# If you hit OOM, change both batch-size values from 64 to 32.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
    --tokenizer_type single \
    --train_file data/processed/mlm/train_single.txt \
    --validation_file data/processed/mlm/val_single.txt \
    --vocab_dir tokenizers/single \
    --output_dir outputs/bert_mlm/single \
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
