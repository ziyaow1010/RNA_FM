#!/bin/bash
# End-to-end smoke test: build tokenizers, prepare a small dataset, and run a
# few training steps for BOTH tokenizers on a single GPU. Fast; verifies the
# whole pipeline runs without OOM or shape errors. NOT a real training run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "[smoke] 1/4  build tokenizers"
echo "==================================================================="
python scripts/build_tokenizers.py

echo "==================================================================="
echo "[smoke] 2/4  prepare small data (10000 records)"
echo "==================================================================="
python scripts/prepare_mlm_data.py \
    --max-records 10000 \
    --max-length 512

# Use a single GPU for the smoke test.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "==================================================================="
echo "[smoke] 3/4  train single (20 steps)"
echo "==================================================================="
python scripts/train_bert_mlm.py \
    --tokenizer_type single \
    --train_file data/processed/mlm/train_single.txt \
    --validation_file data/processed/mlm/val_single.txt \
    --vocab_dir tokenizers/single \
    --output_dir outputs/bert_mlm_smoke/single \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 20 \
    --logging_steps 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8

echo "==================================================================="
echo "[smoke] 4/4  train center3 (20 steps)"
echo "==================================================================="
python scripts/train_bert_mlm.py \
    --tokenizer_type center3 \
    --train_file data/processed/mlm/train_center3.txt \
    --validation_file data/processed/mlm/val_center3.txt \
    --vocab_dir tokenizers/center3 \
    --output_dir outputs/bert_mlm_smoke/center3 \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 20 \
    --logging_steps 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8

echo "==================================================================="
echo "[smoke] DONE. eval results:"
echo "  single : $(cat outputs/bert_mlm_smoke/single/eval_results.json 2>/dev/null)"
echo "  center3: $(cat outputs/bert_mlm_smoke/center3/eval_results.json 2>/dev/null)"
echo "==================================================================="
