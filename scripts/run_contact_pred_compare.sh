#!/bin/bash
# Full supervised contact-prediction comparison on ArchiveII RANDOM split (Part 8).
# 4 FMs, unified frozen-LM + ResNet head. Extract embeddings -> train head -> eval.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
SP=data/contact_eval/splits
EPOCHS=${EPOCHS:-30}

python scripts/make_contact_splits.py

# name : model_dir : model_type : tokenizer_type : vocab_dir
MODELS=(
 "kmer1-BERT:outputs/fm_bert_kmer1:bert:kmer1:tokenizers/single"
 "kmer1-Hybrid:outputs/fm_hybrid_mamba_kmer1:hybrid:kmer1:tokenizers/single"
 "kmer6-BERT:outputs/fm_bert_kmer6:bert:kmer6:tokenizers/kmer6"
 "kmer6-Hybrid:outputs/fm_hybrid_mamba_kmer6:hybrid:kmer6:tokenizers/kmer6"
)
ENTRIES=()
for spec in "${MODELS[@]}"; do
  IFS=: read -r name mdir mtype ttype vdir <<< "$spec"
  echo "==== $name : extract embeddings ===="
  python scripts/extract_lm_embeddings.py --model_dir "$mdir" --model_type "$mtype" \
    --tokenizer_type "$ttype" --vocab_dir "$vdir" --dataset_jsonl data/contact_eval/archiveII.jsonl \
    --model_name "$name" --dataset_name archiveII --layer final --max-len 512 --batch-size 8
  echo "==== $name : train contact head (random split) ===="
  python scripts/train_contact_head.py \
    --embedding_dir outputs/contact_pred/embeddings/$name/archiveII \
    --train_jsonl $SP/archiveII_random_train.jsonl --val_jsonl $SP/archiveII_random_val.jsonl \
    --test_jsonl $SP/archiveII_random_test.jsonl \
    --output_dir outputs/contact_pred/$name/archiveII_random --epochs $EPOCHS --max-len 512 --num-plots 10
  ENTRIES+=( "$name=outputs/contact_pred/$name/archiveII_random" )
done
python scripts/aggregate_contact_pred.py --split archiveII_random \
  --out_prefix outputs/contact_pred/compare_archiveII_random --entries "${ENTRIES[@]}"
echo "[compare] DONE"
