#!/bin/bash
# Download RNAcentral active sequences fasta.gz (~9 GB).
# Target: /beacon-projects/rnallm/data/rnacentral/rnacentral_active.fasta.gz
#SBATCH --job-name=dl_rnacentral
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/download_rnacentral_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/download_rnacentral_%j.err

set -euo pipefail
DEST=/beacon-projects/rnallm/data/rnacentral/rnacentral_active.fasta.gz
mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ]; then
    SIZE=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
    echo "[dl] File exists, size=${SIZE} bytes"
    if [ "$SIZE" -gt 5000000000 ]; then
        echo "[dl] Already downloaded (>5GB), skipping."
        exit 0
    fi
    echo "[dl] File too small, re-downloading..."
fi

URL="https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz"
echo "[dl] Downloading from $URL ..."
curl -C - -L -o "$DEST.tmp" "$URL" && mv "$DEST.tmp" "$DEST"
SIZE=$(stat -c%s "$DEST")
echo "[dl] Done: $DEST  size=$(echo "scale=2; $SIZE/1073741824" | bc) GB"
