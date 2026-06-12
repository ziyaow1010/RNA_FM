#!/bin/bash
# Smoke test: 50 train / 10 val / 10 test, max_len 256, kmer1-BERT, epochs=2.
# Verifies embedding extraction, head forward, loss decrease, metrics, plots.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

SP=data/contact_eval/splits
# build small smoke splits (filter max_len 256)
python - << 'PY'
import json
from pathlib import Path
sp=Path('data/contact_eval/splits')
def take(src,n):
    out=[]
    for l in open(sp/src):
        r=json.loads(l)
        if r['length']<=256: out.append(r)
        if len(out)>=n: break
    return out
tr,va,te=take('archiveII_random_train.jsonl',50),take('archiveII_random_val.jsonl',10),take('archiveII_random_test.jsonl',10)
for name,recs in [('smoke_train',tr),('smoke_val',va),('smoke_test',te)]:
    open(sp/f'{name}.jsonl','w').write('\n'.join(json.dumps(r) for r in recs)+'\n')
open(sp/'smoke_all.jsonl','w').write('\n'.join(json.dumps(r) for r in tr+va+te)+'\n')
print(f'smoke splits: {len(tr)}/{len(va)}/{len(te)}')
PY

echo "=== extract embeddings (kmer1-BERT) ==="
python scripts/extract_lm_embeddings.py --model_dir outputs/fm_bert_kmer1 --model_type bert \
    --tokenizer_type kmer1 --vocab_dir tokenizers/single \
    --dataset_jsonl $SP/smoke_all.jsonl --model_name kmer1-BERT --dataset_name smoke \
    --layer final --max-len 256 --batch-size 8

echo "=== train contact head (epochs=2) ==="
python scripts/train_contact_head.py \
    --embedding_dir outputs/contact_pred/embeddings/kmer1-BERT/smoke \
    --train_jsonl $SP/smoke_train.jsonl --val_jsonl $SP/smoke_val.jsonl --test_jsonl $SP/smoke_test.jsonl \
    --output_dir outputs/contact_pred/smoke/kmer1-BERT --epochs 2 --max-len 256 --num-plots 5

echo "=== smoke test_metrics ==="
cat outputs/contact_pred/smoke/kmer1-BERT/test_metrics.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('F1=%.4f best_F1=%.4f MCC=%.4f AUPRC=%.4f P@L=%.4f'%(d['mean_F1'],d['mean_best_F1'],d['mean_MCC'],d['mean_AUPRC'],d['mean_P_at_L']))"
echo "plots: $(ls outputs/contact_pred/smoke/kmer1-BERT/example_plots/ 2>/dev/null|wc -l) preds: $(ls outputs/contact_pred/smoke/kmer1-BERT/predictions/ 2>/dev/null|wc -l)"
echo "[smoke] DONE"
