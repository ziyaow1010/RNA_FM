#!/bin/bash
# RiNALMo-style ArchiveII leave-one-family-out, supervised contact prediction.
# 4 FMs x 9 held-out families = 36 head trainings (frozen LM, shared ResNet head).
# Parallelized across 8 GPUs. Embeddings reused from the random-split run.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
SP=data/contact_eval/splits
EPOCHS=${EPOCHS:-30}
FAMILIES=${FAMILIES:-"16s 23s 5s grp1 RNaseP srp telomerase tmRNA tRNA"}
NGPU=${NGPU:-8}

# name:model_dir:model_type:tokenizer:vocab_dir
MODELS=(
 "kmer1-BERT:outputs/fm_bert_kmer1:bert:kmer1:tokenizers/single"
 "kmer1-Hybrid:outputs/fm_hybrid_mamba_kmer1:hybrid:kmer1:tokenizers/single"
 "kmer6-BERT:outputs/fm_bert_kmer6:bert:kmer6:tokenizers/kmer6"
 "kmer6-Hybrid:outputs/fm_hybrid_mamba_kmer6:hybrid:kmer6:tokenizers/kmer6"
)

# ensure embeddings exist (reuse; extract only if missing)
for spec in "${MODELS[@]}"; do
  IFS=: read -r name mdir mtype ttype vdir <<< "$spec"
  if [ ! -f outputs/contact_pred/embeddings/$name/metadata.json ]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_lm_embeddings.py --model_dir "$mdir" \
      --model_type "$mtype" --tokenizer_type "$ttype" --vocab_dir "$vdir" \
      --dataset_jsonl data/contact_eval/archiveII.jsonl --model_name "$name" \
      --dataset_name archiveII --layer final --max-len 512 --batch-size 8
  fi
done

# build the 36-job list (one line: name|family)
JOBS=()
for spec in "${MODELS[@]}"; do
  IFS=: read -r name _ _ _ _ <<< "$spec"
  for fam in $FAMILIES; do JOBS+=("$name|$fam"); done
done

# round-robin jobs to GPU workers
run_job () {  # $1=name $2=fam $3=gpu
  local name="$1" fam="$2" gpu="$3"
  local emb=outputs/contact_pred/embeddings/$name/archiveII
  CUDA_VISIBLE_DEVICES=$gpu python scripts/train_contact_head.py --embedding_dir "$emb" \
    --train_jsonl $SP/archiveII_lfo_${fam}_train.jsonl --val_jsonl $SP/archiveII_lfo_${fam}_val.jsonl \
    --test_jsonl $SP/archiveII_lfo_${fam}_test.jsonl \
    --output_dir outputs/contact_pred/$name/archiveII_lfo_${fam} --epochs $EPOCHS --max-len 512 --num-plots 5 \
    > logs/lfo_${name}_${fam}.log 2>&1
}

i=0
for job in "${JOBS[@]}"; do
  gpu=$(( i % NGPU )); IFS='|' read -r name fam <<< "$job"
  ( run_job "$name" "$fam" "$gpu" ) &
  i=$(( i + 1 ))
  # keep at most NGPU jobs in flight
  if (( i % NGPU == 0 )); then wait; fi
done
wait

python scripts/aggregate_contact_lfo.py --families $FAMILIES
echo "[lfo] ALL DONE"
