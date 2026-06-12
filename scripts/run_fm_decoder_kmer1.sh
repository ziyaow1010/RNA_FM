#!/bin/bash
# Decoder-only (GPT2) causal-LM pretraining, kmer1 tokenizer, GPUs 0-3.
# Same data/tokenizer/batch/optimizer/1-epoch as the kmer1 MLM baseline; the
# OBJECTIVE is next-token prediction (not MLM). Sized to match (~4.87M params).
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
DATA=/tmp/rna_fm_data
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29621 \
    scripts/train_decoder_lm.py --tokenizer_type single --max_seq_length 512 \
    --train_file $DATA/train_single.txt --validation_file $DATA/val_single.txt \
    --vocab_dir tokenizers/single --output_dir outputs/fm_decoder_kmer1 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
    --learning_rate 2e-4 --weight_decay 0.01 --warmup_steps 1000 \
    --max_steps 54231 --eval_steps 500 --save_steps 5000 --logging_steps 50 \
    --num_workers 8 --max_eval_samples 2000 --streaming --fp16 \
    > logs/fm_decoder_kmer1.log 2>&1
