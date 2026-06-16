#!/bin/bash
# Hybrid-300M backbone + RiNALMo SS pipeline with RiNALMo's GRADUAL-UNFREEZE
# fine-tuning. ArchiveII leave-one-family-out (RiNALMo fam-fold). 9 families.
# Longest-sequence families first; tiny-sequence tRNA runs last as the 9th.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
EPOCHS=${EPOCHS:-26}
FAMILIES="16s 23s grp1 telomerase srp tmRNA RNaseP 5s tRNA"

i=0
for fam in $FAMILIES; do
  gpu=$(( i % 8 ))
  ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py --family $fam --epochs $EPOCHS \
      > logs/rinalmo_ss_ft_${fam}.log 2>&1 ) &
  i=$(( i + 1 )); (( i % 8 == 0 )) && wait
done
wait

python scripts/rinalmo_ss_ft_aggregate.py
echo "[rinalmo-ss-ft] ALL DONE"
