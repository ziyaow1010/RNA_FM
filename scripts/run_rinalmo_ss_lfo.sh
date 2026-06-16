#!/bin/bash
# Hybrid-300M backbone + RiNALMo secondary-structure pipeline, ArchiveII
# leave-one-family-out (RiNALMo fam-fold splits). 9 families parallel on 8 GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
EPOCHS=${EPOCHS:-25}
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"

# ensure embeddings extracted
[ -d outputs/contact_pred/rinalmo_ss/emb_300m ] || CUDA_VISIBLE_DEVICES=0 python scripts/rinalmo_ss_extract.py

i=0
for fam in $FAMILIES; do
  gpu=$(( i % 8 ))
  ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_train.py --family $fam --epochs $EPOCHS \
      > logs/rinalmo_ss_${fam}.log 2>&1 ) &
  i=$(( i + 1 )); (( i % 8 == 0 )) && wait
done
wait

python scripts/rinalmo_ss_aggregate.py
echo "[rinalmo-ss] ALL DONE"
