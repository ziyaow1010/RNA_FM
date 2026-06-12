#!/bin/bash
# Full unsupervised contact-probing comparison (first version: kmer1 models).
# Models: kmer1-BERT, kmer1-hybrid. Dataset: ArchiveII subset (max_seqs=100,
# max_len=256). Methods: categorical_jacobian + embedding_perturb. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=data/contact_eval/archiveII.jsonl
MAXSEQS=100; MAXLEN=256; BS=64
declare -A MODELS=( ["kmer1-BERT"]="outputs/fm_bert_kmer1:bert"
                    ["kmer1-hybrid"]="outputs/fm_hybrid_mamba_kmer1:hybrid" )

ENTRIES=()
for name in kmer1-BERT kmer1-hybrid; do
  IFS=: read -r mdir mtype <<< "${MODELS[$name]}"
  base="outputs/contact_eval/$name"
  echo "==== $name : categorical_jacobian ===="
  python scripts/extract_contacts_categorical_jacobian.py \
      --model_dir "$mdir" --model_type "$mtype" --tokenizer_type kmer1 \
      --dataset_jsonl "$DATA" --output_dir "$base/categorical_jacobian" \
      --max-seqs $MAXSEQS --max-len $MAXLEN --batch-size $BS --num-plots 10
  echo "==== $name : embedding_perturb ===="
  python scripts/extract_contacts_embedding_perturb.py \
      --model_dir "$mdir" --model_type "$mtype" --tokenizer_type kmer1 \
      --dataset_jsonl "$DATA" --output_dir "$base/embedding_perturb" \
      --max-seqs $MAXSEQS --max-len $MAXLEN --batch-size $BS --num-plots 10 \
      --perturb-layer embedding --mode gaussian --epsilon 1.0
  ENTRIES+=( "$name=$base/categorical_jacobian" "$name=$base/embedding_perturb" )
done

python scripts/aggregate_contact_compare.py --dataset archiveII --entries "${ENTRIES[@]}"
echo "[compare] DONE -> outputs/contact_eval/compare_summary.{csv,json,png}"
