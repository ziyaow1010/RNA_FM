#!/bin/bash
# Sweep non-overlapping k-mer tokenizers k=2,4,5,6 at 100k steps each, same
# setup as single/kmer3. Disk-safe: per-k generate data, clear the HF datasets
# cache before each run, and delete the big train file afterwards (val kept).
# Launch detached (setsid + nohup) so it survives the session disconnecting.
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

for K in 2 4 5 6; do
    echo "[sweep] ===== kmer$K BUILD+DATA $(date -Is) ====="
    python scripts/build_kmer_tokenizer.py --k $K
    python scripts/make_kmer_data.py --k $K
    rm -rf ~/.cache/huggingface/datasets        # keep only current k cached

    echo "[sweep] ===== kmer$K TRAIN START $(date -Is) ====="
    torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
        --tokenizer_type kmer$K \
        --train_file data/processed/mlm/train_kmer$K.txt \
        --validation_file data/processed/mlm/val_kmer$K.txt \
        --vocab_dir tokenizers/kmer$K \
        --output_dir outputs/bert_mlm/kmer${K}_100k \
        $COMMON
    echo "[sweep] ===== kmer$K TRAIN DONE $(date -Is) ====="

    rm -f data/processed/mlm/train_kmer$K.txt   # free disk (val kept for eval)
done

rm -rf ~/.cache/huggingface/datasets
python scripts/plot_compare.py || true
echo "[sweep] ===== ALL DONE $(date -Is) ====="
