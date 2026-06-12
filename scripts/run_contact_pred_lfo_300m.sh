#!/bin/bash
# Wait for the 300M hybrid pretraining to finish, then run the RiNALMo-style
# leave-one-family-out supervised contact-prediction eval on it and compare to
# the 5M kmer1-Hybrid baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm
SP=data/contact_eval/splits
EPOCHS=${EPOCHS:-30}
FAMILIES="16s 23s 5s grp1 RNaseP srp telomerase tmRNA tRNA"
NAME=kmer1-Hybrid-300M
MDIR=outputs/fm_hybrid_mamba_kmer1_300m

echo "[300m-lfo] waiting for 300M training to finish..."
while [ ! -f $MDIR/eval_results.json ]; do sleep 120; done
echo "[300m-lfo] training done. acc=$(python3 -c "import json;print(json.load(open('$MDIR/eval_results.json'))['eval_masked_accuracy'])")"

echo "[300m-lfo] extract embeddings (hidden_dim 1024)"
CUDA_VISIBLE_DEVICES=0 python scripts/extract_lm_embeddings.py --model_dir $MDIR \
  --model_type hybrid --tokenizer_type kmer1 --vocab_dir tokenizers/single \
  --dataset_jsonl data/contact_eval/archiveII.jsonl --model_name $NAME \
  --dataset_name archiveII --layer final --max-len 512 --batch-size 8

emb=outputs/contact_pred/embeddings/$NAME/archiveII
i=0
for fam in $FAMILIES; do
  gpu=$(( i % 8 ))
  ( CUDA_VISIBLE_DEVICES=$gpu python scripts/train_contact_head.py --embedding_dir "$emb" \
      --train_jsonl $SP/archiveII_lfo_${fam}_train.jsonl --val_jsonl $SP/archiveII_lfo_${fam}_val.jsonl \
      --test_jsonl $SP/archiveII_lfo_${fam}_test.jsonl \
      --output_dir outputs/contact_pred/$NAME/archiveII_lfo_${fam} --epochs $EPOCHS --max-len 512 --num-plots 5 \
      > logs/lfo_${NAME}_${fam}.log 2>&1 ) &
  i=$(( i + 1 )); (( i % 8 == 0 )) && wait
done
wait

# macro-average + compare to 5M kmer1-Hybrid
python3 - << 'PY'
import json, statistics
from pathlib import Path
fams="16s 23s 5s grp1 RNaseP srp telomerase tmRNA tRNA".split()
mets=["precision","recall","F1","best_F1","MCC","AUPRC","P_at_L","P_at_L2"]
def macro(model):
    out={}
    for m in mets:
        vals=[]
        for f in fams:
            tm=Path(f"outputs/contact_pred/{model}/archiveII_lfo_{f}/test_metrics.json")
            if tm.exists():
                v=json.load(open(tm)).get(f"mean_{m}")
                if v==v: vals.append(v)
        out[m]=round(statistics.mean(vals),4) if vals else None
    return out
m300=macro("kmer1-Hybrid-300M"); m5=macro("kmer1-Hybrid")
json.dump({"kmer1-Hybrid-300M":m300,"kmer1-Hybrid-5M":m5},
          open("outputs/contact_pred/compare_300m_vs_5m_lfo.json","w"),indent=2)
print("\n=== LFO macro-average: 300M vs 5M (kmer1-Hybrid) ===")
print(f"{'metric':12}{'5M':>10}{'300M':>10}")
for m in mets: print(f"{m:12}{(m5[m] if m5[m] is not None else 0):>10.4f}{(m300[m] if m300[m] is not None else 0):>10.4f}")
PY
echo "[300m-lfo] ALL DONE"
