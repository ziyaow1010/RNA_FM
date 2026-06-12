#!/bin/bash
# Smoke test for the unsupervised contact-probing module: 10 ArchiveII seqs,
# max_len 128, kmer1-BERT, both methods. Confirms score matrices, metrics, and
# plots are produced without OOM. Does NOT run the full eval.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

DATA=data/contact_eval/archiveII.jsonl
MODEL=outputs/fm_bert_kmer1
ROOT=outputs/contact_eval/smoke/kmer1-BERT
COMMON="--model_dir $MODEL --model_type bert --tokenizer_type kmer1 \
  --dataset_jsonl $DATA --max-seqs 10 --max-len 128 --batch-size 64 --num-plots 5"

echo "=== categorical Jacobian ==="
python scripts/extract_contacts_categorical_jacobian.py $COMMON \
    --output_dir $ROOT/categorical_jacobian
echo "=== embedding perturbation ==="
python scripts/extract_contacts_embedding_perturb.py $COMMON \
    --output_dir $ROOT/embedding_perturb --perturb-layer embedding --mode gaussian --epsilon 1.0

echo "=== smoke summaries ==="
for m in categorical_jacobian embedding_perturb; do
  echo "--- $m ---"; cat $ROOT/$m/metrics_summary.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('seqs',d['num_sequences'],'mean_AUPRC=%.3f P@L=%.3f F1=%.3f'%(d['mean_AUPRC'],d['mean_P_at_L'],d['mean_best_F1']))"
  echo "plots: $(ls $ROOT/$m/example_plots/ 2>/dev/null | wc -l)  scores: $(ls $ROOT/$m/scores/ 2>/dev/null | wc -l)"
done
echo "[smoke] DONE"
