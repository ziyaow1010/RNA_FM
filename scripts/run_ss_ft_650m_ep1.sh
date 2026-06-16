#!/bin/bash
# Full gradual-unfreeze RiNALMo SS fine-tune LFO on the Hybrid-650M EPOCH-1
# checkpoint. 9 families across 8 GPUs. Same FT protocol as the 300M comparison
# (epochs chosen so all backbone layers fully unfreeze: 33 layers / 3 per step
# x 3 epochs => unfreeze done by epoch 33, +3 epochs train).
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
export TOKENIZERS_PARALLELISM=false
MODEL=outputs/fm_hybrid_650m/ss_ft_ep1_model
OUT=outputs/fm_hybrid_650m/ss_ft_ep1_lr5e4
EPOCHS=${EPOCHS:-30}
FAMILIES="16s 23s grp1 telomerase srp tmRNA RNaseP 5s tRNA"

i=0
for fam in $FAMILIES; do
  gpu=$(( i % 8 ))
  ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
      --family $fam --model_dir $MODEL --out_dir $OUT --epochs $EPOCHS \
      --base_lr 5e-4 --unfreeze_min_layer 9 \
      > logs/ss_ft_650m_ep1_lr5e4_${fam}.log 2>&1 ) &
  i=$(( i + 1 )); (( i % 8 == 0 )) && wait
done
wait
echo "[ss-ft-650m-ep1] ALL DONE"
