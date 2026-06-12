#!/bin/bash
# Foundation-model pretraining with the Transformer+Mamba HYBRID backbone +
# kmer6 tokenizer. Aligned with the vanilla-BERT kmer6 baseline
# (outputs/bert_mlm/fm_kmer6): same data, tokenizer, max_seq_length(96), batch,
# lr/warmup/wd, 15% MLM, 1 epoch = 54231 steps. ONLY the backbone differs.
#
# NOTE: --tie_word_embeddings is REQUIRED here so the large (15630) MLM-head
# decoder shares the embedding matrix (matches BERT kmer6); untied would add
# ~4M params (+47%). Runs on GPUs 0,1,2,3.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=/tmp/rna_fm_data

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29613 \
    scripts/train_hybrid_mamba_mlm.py \
    --tokenizer_type kmer6 --max_seq_length 96 \
    --train_file $DATA/train_kmer6.txt --validation_file $DATA/val_kmer6.txt \
    --vocab_dir tokenizers/kmer6 \
    --output_dir outputs/fm_hybrid_mamba_kmer6 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
    --learning_rate 2e-4 --weight_decay 0.01 --warmup_steps 1000 \
    --max_steps 54231 --eval_steps 500 --save_steps 5000 --logging_steps 50 \
    --mlm_probability 0.15 --num_workers 8 --max_eval_samples 2000 --streaming \
    --fp16 --tie_word_embeddings \
    --layer_pattern TTMTTM --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 4 \
    > logs/fm_hybrid_mamba_kmer6.log 2>&1
