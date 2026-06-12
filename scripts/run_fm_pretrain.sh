#!/bin/bash
# Foundation-model pretraining: kmer1 (single) and kmer6 trained IN PARALLEL on
# the full <512-base dataset (streaming), identical data/bs/steps, 4 GPUs each.
# A background loop refreshes the combined live chart. Launch detached.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=/tmp/rna_fm_data
# identical training budget; only --max_seq_length differs (single needs up to
# 512 tokens, kmer6 only ~88 = ceil(511/6)+2, so a smaller cap avoids padding waste).
# Scaled up batch (64->256) + workers + lr to use the GPUs (was ~7% memory).
# max_steps = 1 full epoch over the 27,765,912 train seqs at global batch
# 512 (128/device x 4 GPUs) = ceil(27765912/512) = 54231 -> every sequence
# is seen exactly once.
COMMON="--per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
  --learning_rate 2e-4 --weight_decay 0.01 --warmup_steps 1000 --max_steps 54231 \
  --eval_steps 500 --save_steps 5000 --logging_steps 50 --mlm_probability 0.15 \
  --num_workers 8 --max_eval_samples 2000 --streaming"

echo "[fm] START $(date -Is)  kmer1=GPU0-3  kmer6=GPU4-7"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_bert_mlm.py --tokenizer_type single --max_seq_length 512 \
    --train_file $DATA/train_single.txt --validation_file $DATA/val_single.txt \
    --vocab_dir tokenizers/single --output_dir outputs/bert_mlm/fm_single \
    $COMMON > logs/fm_single.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29502 \
    scripts/train_bert_mlm.py --tokenizer_type kmer6 --max_seq_length 96 \
    --train_file $DATA/train_kmer6.txt --validation_file $DATA/val_kmer6.txt \
    --vocab_dir tokenizers/kmer6 --output_dir outputs/bert_mlm/fm_kmer6 \
    $COMMON > logs/fm_kmer6.log 2>&1 &
PID6=$!

# live combined chart while either run is alive
( while kill -0 $PID1 2>/dev/null || kill -0 $PID6 2>/dev/null; do
    python scripts/plot_fm_live.py >/dev/null 2>&1 || true
    sleep 60
  done ) &
PIDP=$!

wait $PID1; echo "[fm] kmer1 (single) done $(date -Is)"
wait $PID6; echo "[fm] kmer6 done $(date -Is)"
kill $PIDP 2>/dev/null || true
python scripts/plot_fm_live.py || true
echo "[fm] ALL DONE $(date -Is)"
