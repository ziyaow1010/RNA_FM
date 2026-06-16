#!/bin/bash
# SS fine-tune LFO on a specific pretrain epoch checkpoint (H200).
# Usage:
#   EPOCH=1 sbatch scripts/ss_ft_h200.sh
#   EPOCH=2 sbatch scripts/ss_ft_h200.sh
#   ...
#
# Each family gets 1 GPU; runs 9 families in 2 waves on 8 GPUs.
# Writes results to outputs/fm_hybrid_650m/ss_eval/epoch{N}/*.json
#SBATCH --job-name=rnafm_ss_ft
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_epoch%a_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_epoch%a_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

EPOCH=${EPOCH:-1}

# Find the epoch checkpoint directory (EpochCheckpointCallback saves here)
EPOCH_CKPT="$CKPT_DIR/epoch_checkpoints/epoch${EPOCH}"

if [ ! -d "$EPOCH_CKPT" ]; then
    # Fall back: pick the latest checkpoint (training may still be in progress)
    EPOCH_CKPT=$(ls -dt "$CKPT_DIR"/checkpoint-* 2>/dev/null | head -1)
    echo "[ss_ft] WARNING: epoch${EPOCH} checkpoint not found, using: $EPOCH_CKPT"
fi

if [ -z "$EPOCH_CKPT" ]; then
    echo "[ss_ft] ERROR: no checkpoint found in $CKPT_DIR" >&2
    exit 1
fi

echo "[ss_ft] Epoch $EPOCH checkpoint: $EPOCH_CKPT"

# Regenerate CT splits if missing (small, <10s)
CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then
    echo "[ss_ft] CT splits missing, regenerating..."
    python scripts/make_ct_splits.py
fi

echo "[ss_ft] Start: $(date)"

OUT="outputs/fm_hybrid_650m/ss_eval/epoch${EPOCH}"
mkdir -p "$OUT" logs

FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % 8 ))
    LOG="logs/ss_ft_epoch${EPOCH}_${fam}.log"
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" \
        --model_dir "$EPOCH_CKPT" \
        --out_dir "$OUT" \
        --epochs 30 \
        --base_lr 5e-4 \
        --unfreeze_min_layer 9 \
        > "$LOG" 2>&1
      echo "[ss_ft] $fam done (gpu $gpu)" ) &
    i=$(( i + 1 ))
    # Launch next wave after 8 jobs
    if (( i % 8 == 0 )); then
        wait
        echo "[ss_ft] Wave $((i/8)) done"
    fi
done
wait

echo "[ss_ft] All families done: $(date)"
echo "[ss_ft] Aggregating results..."

python3 - <<'PYEOF'
import json, statistics
from pathlib import Path
import sys, os

epoch = os.environ.get("EPOCH", "1")
out = Path(f"outputs/fm_hybrid_650m/ss_eval/epoch{epoch}")
rinalmo = {"5s":0.88, "16s":0.74, "23s":0.85, "grp1":0.66,
           "srp":0.70, "telomerase":0.12, "RNaseP":0.80, "tmRNA":0.80, "tRNA":0.93}

rows = []
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf))
    fam = jf.stem
    f1 = d.get("mean_F1", 0.0)
    ref = rinalmo.get(fam, float("nan"))
    rows.append({"family": fam, "hybrid_f1": f1, "rinalmo_f1": ref, "delta": f1-ref})
    print(f"  {fam:12s}  Hybrid={f1:.3f}  RiNALMo={ref:.3f}  Δ={f1-ref:+.3f}")

if rows:
    macro = statistics.mean(r["hybrid_f1"] for r in rows)
    ref_macro = statistics.mean(r["rinalmo_f1"] for r in rows if not (r["rinalmo_f1"] != r["rinalmo_f1"]))
    print(f"\n  {'MACRO':12s}  Hybrid={macro:.3f}  RiNALMo={ref_macro:.3f}  Δ={macro-ref_macro:+.3f}")
    summary = {"epoch": int(epoch), "families": rows, "macro_f1": macro, "rinalmo_macro": ref_macro}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"\n[ss_ft] Summary saved: {out}/summary.json")
PYEOF
