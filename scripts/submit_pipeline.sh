#!/bin/bash
# One-shot pipeline: build_env → download_rna → cluster → pretrain → ss_ft per epoch.
# Usage: bash scripts/submit_pipeline.sh
#
# If data/env already exist the early steps are near-instant (idempotent).
# H200 pretrain is 24h; ss_ft per epoch ~4h each.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== RNA-FM H200 pipeline ==="

# Step 1: Build environment
JID_ENV=$(sbatch --parsable scripts/build_rnafm_env.sh)
echo "[1] build_rnafm_env   → JID=$JID_ENV"

# Step 2: Download RNAcentral (runs in parallel with env build)
JID_DL=$(sbatch --parsable scripts/download_rnacentral.sh)
echo "[2] download_rnacentral → JID=$JID_DL"

# Step 3: Cluster (needs both env + data)
JID_CLUST=$(sbatch --parsable --dependency=afterok:${JID_ENV}:${JID_DL} \
    scripts/cluster_rnacentral.sh)
echo "[3] cluster_rnacentral  → JID=$JID_CLUST (after $JID_ENV + $JID_DL)"

# Step 4: Pretrain 6 epochs on H200 (needs clustering done)
JID_TRAIN=$(sbatch --parsable --dependency=afterok:${JID_CLUST} \
    scripts/pretrain_h200.sh)
echo "[4] pretrain_h200       → JID=$JID_TRAIN (after $JID_CLUST)"

echo ""
echo "=== Submitted. Monitor with: ==="
echo "  squeue -u \$USER"
echo "  tail -f logs/pretrain_h200_*.out"
echo ""
echo "After each epoch checkpoint appears at:"
echo "  /beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch{N}/"
echo "Submit SS eval manually:"
echo "  EPOCH=1 sbatch scripts/ss_ft_h200.sh"
echo "  EPOCH=2 sbatch scripts/ss_ft_h200.sh  # etc."
echo ""
echo "Job chain: $JID_ENV → ($JID_DL ||) → $JID_CLUST → $JID_TRAIN"
