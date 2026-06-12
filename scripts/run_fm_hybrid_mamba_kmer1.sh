#!/bin/bash
# Foundation-model pretraining with the Transformer+Mamba HYBRID backbone,
# kmer1 tokenizer. EVERYTHING is aligned with the vanilla-BERT kmer1 baseline
# (outputs/bert_mlm/fm_single): same data, tokenizer, max_seq_length, batch,
# lr/warmup/wd, 15% MLM, 1 epoch = 54231 steps, 4 GPUs. ONLY the backbone differs.
#
# Runs on GPUs 4,5,6,7. Does NOT kill any existing training.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=/tmp/rna_fm_data

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29611 \
    scripts/train_hybrid_mamba_mlm.py \
    --tokenizer_type single --max_seq_length 512 \
    --train_file $DATA/train_single.txt --validation_file $DATA/val_single.txt \
    --vocab_dir tokenizers/single \
    --output_dir outputs/fm_hybrid_mamba_kmer1 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
    --learning_rate 2e-4 --weight_decay 0.01 --warmup_steps 1000 \
    --max_steps 54231 --eval_steps 500 --save_steps 5000 --logging_steps 50 \
    --mlm_probability 0.15 --num_workers 8 --max_eval_samples 2000 --streaming \
    --fp16 \
    --layer_pattern TTMTTM --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 4 \
    > logs/fm_hybrid_mamba_kmer1.log 2>&1
