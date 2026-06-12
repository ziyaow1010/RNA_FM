#!/bin/bash
# 300M Transformer+Mamba HYBRID, kmer1, full 1 epoch. hidden 1024 / 24 layers
# (TTM x8) / 16 heads / intermediate 4096 / mamba_expand 4 = ~309.8M params.
# 8 GPUs, bs 32/device (global 256), bf16, 1 epoch = 108461 steps.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
DATA=/tmp/rna_fm_data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29701 \
    scripts/train_hybrid_mamba_mlm.py --tokenizer_type single --max_seq_length 512 \
    --train_file $DATA/train_single.txt --validation_file $DATA/val_single.txt \
    --vocab_dir tokenizers/single --output_dir outputs/fm_hybrid_mamba_kmer1_300m \
    --hidden_size 1024 --num_attention_heads 16 --intermediate_size 4096 \
    --layer_pattern TTMTTMTTMTTMTTMTTMTTMTTM --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 4 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 --weight_decay 0.01 --warmup_steps 3000 --max_steps 108461 \
    --max_grad_norm 0.5 \
    --eval_steps 1000 --save_steps 5000 --logging_steps 50 --mlm_probability 0.15 \
    --num_workers 8 --max_eval_samples 2000 --streaming --bf16 \
    > logs/fm_hybrid300m_kmer1.log 2>&1
