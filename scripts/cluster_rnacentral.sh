#!/bin/bash
# Cluster RNAcentral to ~17M representative sequences (matching RiNALMo pretraining).
# Uses mmseqs2 easy-cluster; installs it via conda if not found.
# Input:  /beacon-projects/rnallm/data/rnacentral/rnacentral_active.fasta.gz
# Output: /beacon-projects/rnallm/data/rnacentral/clustered_17m.fasta
#SBATCH --job-name=cluster_rna
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/cluster_rnacentral_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/cluster_rnacentral_%j.err

set -euo pipefail
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA=/beacon-projects/rnallm/data/rnacentral
INPUT=$DATA/rnacentral_active.fasta.gz
OUT_FASTA=$DATA/clustered_17m.fasta
TMP=$DATA/mmseqs_tmp
CLUST_PREFIX=$DATA/mmseqs_clust

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

# mmseqs should have been installed in build_env step
if ! command -v mmseqs &>/dev/null; then
    # Fallback: download static binary
    echo "[cluster] mmseqs not found, downloading static binary..."
    MMSEQS_BIN=$DATA/mmseqs_bin
    mkdir -p $MMSEQS_BIN
    curl -L -o $MMSEQS_BIN/mmseqs.tar.gz \
        "https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz"
    tar -xf $MMSEQS_BIN/mmseqs.tar.gz -C $MMSEQS_BIN --strip-components=1
    export PATH="$MMSEQS_BIN/bin:$PATH"
fi

echo "[cluster] mmseqs version: $(mmseqs --version 2>/dev/null || echo unknown)"

# Decompress to temp fasta for mmseqs
PLAIN_FASTA=$DATA/rnacentral_active.fasta
if [ ! -f "$PLAIN_FASTA" ]; then
    echo "[cluster] Decompressing fasta..."
    pigz -d -k -p 8 "$INPUT" 2>/dev/null || gzip -d -k "$INPUT"
fi

echo "[cluster] Input sequences: $(grep -c '^>' $PLAIN_FASTA)"

mkdir -p "$TMP"
echo "[cluster] Running mmseqs easy-linclust (seq-id 0.8)..."
# Use linclust (faster O(N) algorithm) targeting ~17M reps from 40.7M
# --min-seq-id 0.8 is a common threshold that reduces ~40% redundancy
mmseqs easy-linclust \
    "$PLAIN_FASTA" \
    "$CLUST_PREFIX" \
    "$TMP" \
    --min-seq-id 0.8 \
    --cov-mode 1 \
    -c 0.8 \
    --threads 32 \
    -v 1

REP_FASTA="${CLUST_PREFIX}_rep_seq.fasta"
N_REPS=$(grep -c '^>' "$REP_FASTA")
echo "[cluster] Representatives: $N_REPS"
cp "$REP_FASTA" "$OUT_FASTA"

echo "[cluster] Done: $OUT_FASTA  ($N_REPS sequences)"
echo "[cluster] If count >> 17M, re-run with higher --min-seq-id (e.g. 0.9)"
echo "[cluster] If count << 17M, re-run with lower  --min-seq-id (e.g. 0.7)"
