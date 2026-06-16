# RNA Hybrid Foundation Model (Transformer + Mamba) — vs RiNALMo-Giga

> **Current focus → read [`HANDOFF.md`](HANDOFF.md).** It has the full state, verified
> faithfulness checklist, key findings, and exact run commands to continue on a bigger box (H200).

We pretrain an **RNA Hybrid (Transformer+Mamba) foundation model** (`HybridMambaForMaskedLM`,
**664M** params) on full RNAcentral and compare it to **RiNALMo-Giga (650M)** on the **ArchiveII
leave-one-family-out** secondary-structure benchmark, using RiNALMo's **exact official downstream
pipeline** (head / BCE-on-upper-triangle loss / canonical+greedy decoder / flexible-F1 /
per-fold threshold tuning / gradual-unfreeze fine-tune). Only the **backbone** differs.

**Validated:** our eval pipeline reproduces RiNALMo's *own* released giga 5s weights at **F1 0.882**
(paper 0.88), and our decoder+metric are bit-identical to theirs — so the eval is not the variable.
Pretraining ran ~1/6 epochs on 8×L20 before moving to H200; see `HANDOFF.md` to finish it.

Pretraining/eval entry points: `scripts/train_hybrid650_mlm.py`, `scripts/rna_stream_dataset.py`,
`scripts/run_hybrid650.sh`, `scripts/rinalmo_ss_finetune.py`, `scripts/run_ss_ft_650m_ep1.sh`,
`scripts/eval_rinalmo_weights.py`.

**Load the model** (downloads the epoch-1 checkpoint from
[`Ziyao1010/RNA_FM`](https://huggingface.co/Ziyao1010/RNA_FM) and runs a forward pass):
```bash
python scripts/load_hybrid650.py                 # or --model_dir <local> / --seq ACGU...
```
```python
from scripts.load_hybrid650 import load_hybrid, encode
import torch
model, _ = load_hybrid()                         # HybridMambaForMaskedLM (664M), eval mode
ids = torch.tensor([encode("GGGCUAUUAGCUCAGUUGG")]).cuda()
out = model(input_ids=ids, attention_mask=torch.ones_like(ids))   # out.logits: [1, L+2, 10]
```

---

# RNAcentral Download & Analysis Pipeline

A small, self-contained pipeline that downloads the
[RNAcentral](https://rnacentral.org/) *active* sequence set and computes
streaming statistics over the FASTA file (sequence counts, length
distribution, alphabet composition, duplicates, taxonomy breakdown, etc.).

The analysis is **streaming**: the FASTA file is read one record at a time and
the full file is never loaded into memory, so the same script runs on a laptop
sample or the full multi-gigabyte release on a cluster.

## Project layout

```
project_root/
├── environment.yml                       # conda environment (rnafm)
├── README.md
├── data/
│   └── raw/
│       └── rnacentral_active.fasta.gz    # downloaded automatically if missing
├── outputs/
│   └── rnacentral_stats/                 # analysis outputs
├── logs/                                 # slurm logs
└── scripts/
    ├── analyze_rnacentral.py
    └── run_analyze_rnacentral.slurm
```

## Environment setup

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate rnafm
```

Update the environment (after editing `environment.yml`):

```bash
conda env update -f environment.yml --prune
```

The `rnafm` environment uses Python 3.11 with `pip`, `tqdm`, `pandas`, `numpy`.

## Data download

The target file is:

```
https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
```

If `data/raw/rnacentral_active.fasta.gz` does not exist, the analysis script
downloads it automatically on first run. (It is several GB compressed, so the
first run can take a while.)

## Running the analysis

Run a small sample (fast, for testing):

```bash
python scripts/analyze_rnacentral.py \
    --max-records 10000
```

Run the full dataset:

```bash
python scripts/analyze_rnacentral.py
```

### Options

| Option           | Description                                                         |
|------------------|---------------------------------------------------------------------|
| `--input`        | Path to a `.fasta`, `.fa`, or `.fasta.gz` file. Defaults to `data/raw/rnacentral_active.fasta.gz` (auto-downloaded). |
| `--max-records`  | Stop after N sequences. Omit to process the entire file.            |
| `--output-dir`   | Output directory. Defaults to `outputs/rnacentral_stats`.           |

## Output files

All outputs go to `outputs/rnacentral_stats/` by default.

| File                   | Contents                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `summary.json`         | All headline statistics: sequence count, nucleotide total, length stats (min/max/mean/median/p50/p75/p90/p95/p99), alphabet counts (A/U/T/C/G/N/other), ambiguous-nucleotide ratio, duplicate exact-sequence count, top 20 longest sequences, top 20 taxonomy IDs. |
| `length_hist.csv`      | Length histogram as `bin_start,bin_end,count` (50 bins).                  |
| `alphabet_counts.csv`  | Per-symbol counts as `symbol,count` for A, U, T, C, G, N, other.         |
| `sample_records.jsonl` | First 100 records, one JSON object per line: `id`, `taxid`, `length`, `sequence_prefix`. |

### Header parsing

Taxonomy IDs are parsed from the species-specific RNAcentral header format,
e.g.:

```
>URS00005EB5B7_9606
```

The part after the underscore (`9606`) is the NCBI taxonomy ID.

> **Note on the active release.** The headers in
> `rnacentral_active.fasta.gz` are bare URS accessions
> (`>URS000149A9AF rRNA from 1 species`) and do **not** carry a `_taxid`
> suffix — that suffix only appears in the species-specific id files. So
> against the active file `top_taxonomy_ids` is empty; the parser still
> extracts taxids automatically when run on a species-specific FASTA.

### Alphabet (T vs U)

The active FASTA stores sequences in the **DNA alphabet** (`T`), not `U`.
The analyzer tracks `A, U, T, C, G, N` and treats all of `A/U/T/C/G` as
definite (non-ambiguous) bases, so the ambiguous-nucleotide ratio reflects
only `N` and IUPAC ambiguity codes (`R, Y, S, W, ...`), which land in
`other`.

## Running on the Beacon cluster (SLURM)

A batch script is provided at `scripts/run_analyze_rnacentral.slurm`:

- **Job name:** `rnacentral_stats`
- **CPUs:** 8
- **Memory:** 64G
- **Walltime:** 12:00:00
- **Logs:** written to `logs/`

Submit the job:

```bash
sbatch scripts/run_analyze_rnacentral.slurm
```

Check job status:

```bash
squeue -u $USER
```

Cancel a job:

```bash
scancel JOBID
```

## Data compliance

RNAcentral is a **public** database, so the sequence data itself is freely
distributable.

The Beacon cluster does **not** permit the following on it:

- **PHI** (Protected Health Information)
- **PII** (Personally Identifiable Information)
- **HIPAA**-regulated data
- **CUI** (Controlled Unclassified Information)

Before uploading **any** dataset to the cluster, submit a **dataset review**
following the **UM-IHC** process. This applies even when you believe the data
is public — the review confirms that no restricted data classes are present.

---

# Basic BERT MLM Pretraining

A minimal, vanilla BERT masked-language-model (MLM) pretraining setup to
compare nucleotide tokenizers on an identical model architecture:

1. **single** — one token per nucleotide (`A U C G N`)
2. **kmer{k}** — one **non-overlapping** k-mer token per k bases (the variant
   actually used for the experiments; k = 2..6)
3. **center3** — one **overlapping** centered 3-mer per nucleotide — kept only
   to demonstrate a masking leakage failure mode (see results); do not use it.

> **The experiments, full results, model paths, and conclusions are in the
> [Experiments & Results](#experiments--results--tokenizer-comparison-single-vs-non-overlapping-k-mer)
> section at the end of this file.** This section documents the basic pipeline.

This is plain MLM only. There is **no** RNA structure, pairing bias,
contrastive loss, span masking, custom biological loss, learned tokenizer,
LoRA, MoE, or Mamba. The `BertConfig` is identical for both runs; only the
tokenizer/vocab (and therefore `vocab_size`) differs.

Model: `hidden_size=256`, `num_hidden_layers=6`, `num_attention_heads=4`,
`intermediate_size=1024`, `max_position_embeddings=514`, `type_vocab_size=1`,
dropout `0.1`. MLM with `DataCollatorForLanguageModeling`,
`mlm_probability=0.15`.

## 1. Environment install

The training stack needs PyTorch + HuggingFace. Update the conda env:

```bash
conda env update -f environment.yml --prune
conda activate rnafm
```

If conda's CUDA PyTorch resolution is uncertain on your machine, install with
pip instead (inside the activated env):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" "datasets==3.1.0" \
    "accelerate>=0.34,<1.2" "tokenizers>=0.20,<0.21" matplotlib numpy tqdm pandas
```

> **Version note:** stay on the **transformers 4.x** line. transformers 5.x's
> slow tokenizer loader silently drops plain `vocab.txt` tokens (every base
> becomes `[UNK]`), which breaks the char/k-mer vocabularies used here.
> (For a different CUDA build see https://pytorch.org/get-started/locally/.)

## 2. Build tokenizers

```bash
python scripts/build_tokenizers.py
```

Produces:

| File                          | Vocab size | Contents                                  |
|-------------------------------|-----------|--------------------------------------------|
| `tokenizers/single/vocab.txt` | 10        | 5 specials + `A U C G N`                    |
| `tokenizers/center3/vocab.txt`| 130       | 5 specials + all 5³ = 125 3-mers over `AUCGN` |

Both are loadable by `BertTokenizerFast` / `BertTokenizer`.

## 3. Preprocess data

```bash
python scripts/prepare_mlm_data.py            # full data
python scripts/prepare_mlm_data.py --max-records 10000 --max-length 512   # small
```

Streaming read of `data/raw/rnacentral_active.fasta.gz`. Per sequence:
uppercase → `T→U` → any non-`AUCG` char → `N`. Sequences longer than
`--max-length` are split into non-overlapping chunks. Writes parallel
train/val files for both tokenizers (the same chunks, same split):

```
data/processed/mlm/train_single.txt   val_single.txt
data/processed/mlm/train_center3.txt  val_center3.txt
data/processed/mlm/data_stats.json
```

Key options: `--input --output-dir --max-records --val-ratio (0.01)
--max-length (512) --min-length (20) --seed (42)`.

## 4. Smoke test

Runs the whole pipeline (build → prepare 10k records → 20 training steps for
each tokenizer) on a single GPU:

```bash
bash scripts/run_smoke_test.sh
```

## 5. 8-GPU training — single tokenizer

```bash
bash scripts/run_pretrain_single_8gpu.sh
```

which runs:

```bash
torchrun --nproc_per_node=8 scripts/train_bert_mlm.py \
    --tokenizer_type single \
    --train_file data/processed/mlm/train_single.txt \
    --validation_file data/processed/mlm/val_single.txt \
    --vocab_dir tokenizers/single \
    --output_dir outputs/bert_mlm/single \
    --max_seq_length 512 --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 --learning_rate 1e-4 \
    --weight_decay 0.01 --warmup_steps 1000 --max_steps 20000 \
    --eval_steps 1000 --save_steps 5000 --logging_steps 50 \
    --mlm_probability 0.15 --num_workers 8
```

## 6. 8-GPU training — center3 tokenizer

```bash
bash scripts/run_pretrain_center3_8gpu.sh
```

Identical command with `--tokenizer_type center3` and the `center3` files /
vocab dir and `--output_dir outputs/bert_mlm/center3`.

> **OOM?** Lower `--per_device_train_batch_size` and
> `--per_device_eval_batch_size` from `64` to `32` in the launch script.

## Live monitoring (masked-token success rate)

During training, every `--eval_steps` the model is evaluated on a small fixed
validation subset (`--max_eval_samples`, default 2000 sequences) and the
**masked-token prediction success rate** is computed — i.e. of the ~15% of
tokens that the MLM collator masks out, the fraction the model predicts
correctly. Each evaluation refreshes, in the run's `output_dir`:

| File                    | Contents                                                  |
|-------------------------|-----------------------------------------------------------|
| `live_metrics.png`      | Auto-updating chart: loss (train+eval) and masked accuracy |
| `eval_metrics.csv`      | `step,eval_loss,masked_accuracy` — one row per eval        |
| `metrics_history.json`  | Full train-loss and eval series                            |

Watch it live (the PNG re-renders in place; reopen / use an image viewer that
auto-reloads):

```bash
watch -n10 'tail -n +1 outputs/bert_mlm/single/eval_metrics.csv'
# the chart: outputs/bert_mlm/single/live_metrics.png
```

## 7. Viewing loss / eval loss

- **Live**: training and eval loss print to stdout every `--logging_steps` /
  `--eval_steps`. Redirect to a log file if running detached:
  `bash scripts/run_pretrain_single_8gpu.sh 2>&1 | tee logs/single.log`.
- **After**: each run's `output_dir` contains
  - `trainer_state.json` — full `log_history` (every logged train loss + eval loss vs step)
  - `eval_results.json` — final `eval_loss` and `eval_perplexity`

Quick peek:

```bash
python -c "import json; s=json.load(open('outputs/bert_mlm/single/trainer_state.json')); \
print([(h['step'], h.get('eval_loss')) for h in s['log_history'] if 'eval_loss' in h])"
```

## 8. Comparing the two tokenizers

Train both with identical hyperparameters (the provided launch scripts already
match), then compare the final validation MLM loss / perplexity:

```bash
echo  single:; cat outputs/bert_mlm/single/eval_results.json
echo center3:; cat outputs/bert_mlm/center3/eval_results.json
```

Lower `eval_loss` / `eval_perplexity` = the model predicts masked tokens
better under that tokenization. **Caveat:** the loss is computed over different
vocabularies (10 vs 130 classes), so MLM loss is **not** directly comparable as
an absolute number — a 3-mer prediction is intrinsically harder (125-way vs
5-way). Use it as a sanity signal and judge the tokenizers mainly via
downstream task performance in a later stage; within a single tokenizer, the
loss/perplexity curve is the right convergence indicator.

---

# Experiments & Results — tokenizer comparison (single vs non-overlapping k-mer)

This section records the actual experiments run on this project and where the
artifacts live. The question: **for RNA MLM pretraining, is a single-base
tokenizer or a non-overlapping k-mer tokenizer better, and what k?**

## Setup (identical for every run)

- **Data**: first 1,000,000 RNAcentral active records → **1,921,486** train /
  **19,331** val chunks (≤512 bases, mean 388), same chunks & split for all
  tokenizers (`data/processed/mlm/{train,val}_single.txt` is the canonical
  source; k-mer text is derived from it via `scripts/make_kmer_data.py`).
- **Model** (identical except vocab): `BertForMaskedLM`, hidden 256, 6 layers,
  4 heads, intermediate 1024, max_pos 514, dropout 0.1. Plain MLM, 15% masking.
- **Training**: 100,000 steps, global batch 512 (64 × 8 GPU), lr 1e-4, warmup
  1000, AdamW wd 0.01, fp16. Drivers: `scripts/run_5x_experiments.sh`
  (single, kmer3) and `scripts/run_kmer_sweep.sh` (kmer2/4/5/6).

## Trained models (final, 100k steps)

| tokenizer | k | vocab | model path | ~train time (8×L20) |
|-----------|---|-------|------------|----------------------|
| single  | 1 | 10     | `outputs/bert_mlm/single_100k/` | ~3.4 h* |
| kmer2   | 2 | 30     | `outputs/bert_mlm/kmer2_100k/`  | ~2.2 h |
| kmer3   | 3 | 130    | `outputs/bert_mlm/kmer3_100k/`  | ~1.8 h |
| kmer4   | 4 | 630    | `outputs/bert_mlm/kmer4_100k/`  | ~1.7 h |
| kmer5   | 5 | 3,130  | `outputs/bert_mlm/kmer5_100k/`  | ~1.8 h |
| kmer6   | 6 | 15,630 | `outputs/bert_mlm/kmer6_100k/`  | ~2.5 h |

Each dir holds the final model (`model.safetensors` + `config.json` +
`vocab.txt`), checkpoints, `trainer_state.json`, `eval_results.json`, and the
live training curve `live_metrics.png` / `eval_metrics.csv`.
*single's time was inflated by GPU sharing; on a free node ~1.5–2 h is typical.

> **k-mer = 1 nominal vocab note.** Non-overlapping k-mer tokenizers reuse the
> same alphabet {A,U,C,G,N}; vocab = 5^k + 5 special tokens. Tokenizers live in
> `tokenizers/kmer{k}/`. The **overlapping** "center3" tokenizer is kept only to
> reproduce the leakage finding below — do **not** use it for real comparisons.

## Why metrics must be normalized to per-base

The tokenizers predict over different label spaces (5 vs 5^k classes), so raw
token accuracy / perplexity are **not comparable**. Everything is unified to a
**per-base** metric:

- **per-base accuracy**: decode each predicted k-mer to k bases, compare base
  by base to the truth.
- **per-base NLL / perplexity**: `L_base = L_token / k`, `PPL_base = exp(L_base)`.
  Random baselines then coincide at `PPL_base = exp(log 5) = 5` for every k.

Eval is reproducible via `scripts/eval_kmer_perbase.py` (per-base acc + PPL for
one model/k) and `scripts/eval_aligned_span.py` (strict cross-k alignment).

## Result 1 — k-mer sweep, native MLM masking

Each model masks 15% of its own tokens. Validation set (19,331 seqs), seed 42.
Figure: `outputs/bert_mlm/kmer_sweep_summary.png`, data:
`outputs/bert_mlm/kmer_sweep_summary.json`.

| model  | k | token_acc (not comparable) | **per-base acc** | PPL_token (not comp.) | **PPL_base** |
|--------|---|----------------------------|------------------|------------------------|--------------|
| single | 1 | 57.1% | 57.1% | 2.58  | 2.580 |
| kmer2  | 2 | 53.1% | **65.7%** | 4.50  | 2.120 |
| kmer3  | 3 | 47.0% | 65.3% | 9.27  | 2.101 |
| kmer4  | 4 | 43.6% | 65.0% | 19.73 | 2.107 |
| kmer5  | 5 | 41.7% | 64.9% | 38.56 | 2.076 |
| kmer6  | 6 | 41.1% | 65.2% | 75.04 | **2.054** |
| random | — | — | 20% | — | 5.00 |

**Findings:** any non-overlapping k-mer (k≥2) beats single base on both per-base
accuracy (~65% vs 57%) and per-base perplexity (~2.1 vs 2.58). Accuracy
**saturates at k≥2**; PPL_base keeps improving slightly with k (best at k=6).
Token-level metrics degrade monotonically with k purely from the class-count
explosion — exactly why per-base normalization is required. No collapse even at
k=6 (vocab 15,630) at this data scale.

## Result 2 — strictly mask-aligned span reconstruction

To hide the **identical bases** for every k, mask in `lcm(1..6)=60`-base blocks
and have each model mask whole tokens covering them (`scripts/eval_aligned_span.py`).
Tested span lengths 60/120/180/240. Figures:
`outputs/bert_mlm/aligned_vs_native.png`, `outputs/bert_mlm/aligned_span_sweep.png`;
data: `outputs/bert_mlm/aligned_span_{60,120,180,240}.json`.

**PPL_base** (lower better), rows = masked-span length, cols = k:

| span\k | 1 (single) | 2 | 3 | 4 | 5 | 6 |
|--------|------------|---|---|---|---|---|
| 60  | 3.373 | 3.311 | 3.292 | 3.048 | 2.727 | **2.552** |
| 120 | 3.780 | 3.653 | 3.761 | 3.494 | 3.113 | **2.880** |
| 180 | 4.099 | 3.883 | 4.162 | 3.704 | 3.313 | **3.050** |
| 240 | 4.850 | 4.071 | 4.655 | 4.122 | 3.581 | **3.355** |

**Findings:** when all k face the *same* hard hole, **larger k is clearly better
and the advantage grows with span length** (single↔kmer6 PPL gap widens from
0.82 at span 60 to 1.49 at span 240). Filling long holes rewards multi-base
joint modeling, which larger k-mers provide; single can only predict bases
independently. A mild non-monotonic dip appears at k=3/4 before k=5/6 pull ahead.

## Bottom line

- **Non-overlapping k-mer (k≥2) > single base**, confirmed by per-base accuracy,
  per-base perplexity, and the strict mask-aligned test.
- **Best k depends on the task**: short-context MLM **saturates at k=2/3**
  (k=3 is the sweet spot — small vocab, short sequences, fast); long-range /
  long-span reconstruction **keeps favoring larger k** (k=6 best).
- These are intrinsic MLM metrics; a **downstream task** (e.g. RNA family
  classification, secondary structure) is the definitive next step.

## ⚠️ Leakage caveat: do NOT use overlapping (centered) k-mers with single-token MLM

An overlapping centered-3-mer tokenizer (token i = bases i−1,i,i+1) shares all
of a masked token's bases with its unmasked neighbors, so the model trivially
copies the answer. Its masked accuracy spikes to ~96% (vs ~65% leak-free) — a
pure artifact. Always use **non-overlapping** k-mers, or mask whole spans.
(`build_tokenizers.py` still builds `center3`/`run_pretrain_center3_8gpu.sh`
only to reproduce this finding.)

## Result artifacts index

| file | what |
|------|------|
| `outputs/bert_mlm/compare_acc.png` | training accuracy curves vs step (all 6 models) |
| `outputs/bert_mlm/kmer_sweep_summary.{png,json}` | per-base acc & PPL vs k (native masking) |
| `outputs/bert_mlm/aligned_vs_native.png` | native vs aligned-60 per-base acc/PPL vs k |
| `outputs/bert_mlm/aligned_span_sweep.png` | per-base acc/PPL vs k for spans 60/120/180/240 |
| `outputs/bert_mlm/aligned_span_{60,120,180,240}.json` | raw aligned-span numbers |

## Reproducing the evaluation

```bash
conda activate rnafm
# per-base accuracy + per-base perplexity for one model
python scripts/eval_kmer_perbase.py --model_dir outputs/bert_mlm/kmer3_100k --k 3
# strict cross-k aligned-span comparison (all models), e.g. 120-base spans
python scripts/eval_aligned_span.py --block_size 120 --out outputs/bert_mlm/aligned_span_120.json
# regenerate the summary figures
python scripts/plot_compare.py
```

---

# Transformer+Mamba hybrid backbone (kmer1)

A backbone-only variant of the kmer1 BERT MLM baseline: the encoder is a mix of
Transformer and Mamba layers, **everything else identical** (tokenizer, data,
max_seq_length, MLM objective + 15% masking, optimizer/lr/warmup/wd, batch, 1
epoch). Goal: does a Transformer+Mamba hybrid beat vanilla BERT under the same
kmer1 setup.

## Install mamba-ssm (do this carefully)

`pip install mamba-ssm` from PyPI **upgrades torch to a cu13 build and breaks
the env**. Install the prebuilt wheels matching this env (torch 2.5.1+cu121,
cp311, cxx11abi=False) with `--no-deps`:

```bash
conda activate rnafm
pip install --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install --no-deps \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('mamba ok')"
```

If it was already mis-installed (torch upgraded), restore with:
`pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121`,
remove any `nvidia-*-cu13` packages, and reinstall
`nvidia-nccl-cu12==2.21.5 nvidia-cudnn-cu12==9.1.0.70 triton==3.1.0` (`--no-deps`).

## Model

`models/hybrid_mamba_bert.py` → `HybridMambaForMaskedLM`:
- **Embeddings / MLM head**: reuse HuggingFace `BertEmbeddings` / `BertOnlyMLMHead`
  (identical to the BERT baseline). MLM head decoder is **untied** by default.
- **Encoder**: layer pattern `TTMTTM` (6 layers = 4 `BertLayer` Transformer +
  2 Mamba). Transformer block = `BertLayer` (**post-LN**, gelu, same
  hidden/heads/intermediate/dropout as baseline, uses attention_mask). Mamba
  block = **pre-LN residual** `x + dropout(Mamba(LayerNorm(x)))` using
  `mamba_ssm.modules.mamba_simple.Mamba` (d_state 16, d_conv 4, **expand 4**).
- **Padding limitation**: Mamba is **causal** and does NOT consume the
  attention_mask. Padding is right-side and Mamba scans left→right, so trailing
  pads cannot affect earlier real-token states; pad positions carry labels=-100
  so never enter the loss. Bidirectional context comes from the Transformer layers.

## Parameter match (`scripts/count_model_params.py` → `outputs/model_param_compare.json`)

| backbone | embedding | encoder | MLM head | total |
|----------|-----------|---------|----------|-------|
| vanilla BERT  | 134,912 | 4,738,560 | 66,314 | **4,939,786** |
| hybrid (expand 4) | 134,912 | 4,911,104 | 68,874 | **5,114,890** |

Total diff **+3.54%** (within ±10%). `mamba_expand` is the param-match knob
(expand 3 → −5.32%, expand 4 → +3.54%; 4 is closest).

## Smoke test (does NOT start the real run)

```bash
bash scripts/run_hybrid_smoke_test.sh   # param count + 20 steps + eval on GPUs 4,5
```

## Full 1-epoch pretraining (GPUs 4,5,6,7)

```bash
bash scripts/run_fm_hybrid_mamba_kmer1.sh
```
Aligned with the BERT baseline (`outputs/bert_mlm/fm_single`): bs 128/device ×
4 GPU, lr 2e-4, warmup 1000, wd 0.01, max_seq 512, 15% MLM, streaming,
`--max_steps 54231` (1 epoch). Output → `outputs/fm_hybrid_mamba_kmer1/`.

## Backbone comparison

```bash
python scripts/compare_fm_backbones.py    # after both backbones finish
```
Same val set, same mask seed, same 15% masking. Writes
`outputs/fm_backbone_compare.{json,png}` (eval loss, token=per-base accuracy,
per-base NLL/PPL, train runtime, tokens/sec, peak eval GPU mem, param count).
The BERT baseline is read from `outputs/fm_bert_kmer1/` (falls back to
`outputs/bert_mlm/fm_single`).

---

# Foundation models: tokenizer × backbone × objective (full 28M dataset)

Six RNA foundation models trained on the **full** RNAcentral active set (all
sequences < 512 bases, **27.77M** train / 139k val, 7.14B bases, streamed),
**1 epoch each** under identical hyperparameters — crossing three axes:

- **tokenizer**: kmer1 (single base) · kmer6 (non-overlapping 6-mer)
- **backbone**: vanilla BERT encoder · Transformer+Mamba hybrid encoder · GPT-2 decoder
- **objective**: MLM (15% masking) for the encoders · causal next-token for the decoder

Only the crossed axis changes; data, batch (128/device × 4 GPU = 512 global),
lr 2e-4, warmup 1000, wd 0.01, 1 epoch (54,231 steps), fixed-pad streaming are
all identical. Parameter counts are matched within ±10% across backbones for
each tokenizer.

## The 6 models

| model | tokenizer | backbone | objective | params | path |
|-------|-----------|----------|-----------|--------|------|
| kmer1-BERT    | kmer1 | BERT encoder   | MLM    | 4,939,786 | `outputs/bert_mlm/fm_single/` (alias `outputs/fm_bert_kmer1`) |
| kmer1-hybrid  | kmer1 | T+Mamba hybrid | MLM    | 5,114,890 | `outputs/fm_hybrid_mamba_kmer1/` |
| kmer1-decoder | kmer1 | GPT-2 decoder  | causal | 4,872,704 | `outputs/fm_decoder_kmer1/` |
| kmer6-BERT    | kmer6 | BERT encoder   | MLM    | 8,954,126 | `outputs/bert_mlm/fm_kmer6/` |
| kmer6-hybrid  | kmer6 | T+Mamba hybrid | MLM    | 9,126,670 | `outputs/fm_hybrid_mamba_kmer6/` |
| kmer6-decoder | kmer6 | GPT-2 decoder  | causal | 8,764,928 | `outputs/fm_decoder_kmer6/` |

(All share dims hidden 256 / 6 layers / 4 heads / intermediate 1024. Hybrid =
pattern `TTMTTM` (4 Transformer + 2 Mamba), mamba_expand 4; kmer6 hybrid &
decoder tie the lm-head to embeddings so the 15,630-vocab projection isn't
doubled. Each dir holds final model, checkpoints, `eval_results.json`,
`trainer_state.json`, `model_config.json`, `live_metrics.png`.)

## Data prep & training scripts

```bash
# Build the full <512-base dataset (streamed, single + kmer6 text on /tmp)
python scripts/prepare_fm_data.py --output-dir /tmp/rna_fm_data --kmers 1 6
# (measure-only size first: python scripts/prepare_fm_data.py --measure-only)

# MLM encoders (single + kmer6 in parallel, 4 GPUs each): BERT & hybrid
bash scripts/run_fm_pretrain.sh                 # BERT kmer1 + kmer6
bash scripts/run_fm_hybrid_mamba_kmer1.sh       # hybrid kmer1 (GPU 4-7)
bash scripts/run_fm_hybrid_mamba_kmer6.sh       # hybrid kmer6 (GPU 0-3)

# Decoder-only (GPT-2, next-token)
bash scripts/run_fm_decoder_kmer1.sh            # GPU 0-3
bash scripts/run_fm_decoder_kmer6.sh            # GPU 4-7
```

Training scripts: `train_bert_mlm.py` (BERT MLM, `--streaming`),
`train_hybrid_mamba_mlm.py` (hybrid MLM), `train_decoder_lm.py` (GPT-2 causal).
All reuse one streaming pipeline + live-chart callback; fixed padding
(`pad_to_multiple_of=max_seq_length`) is required for DDP + streaming.

## Cross-model results (1 epoch, same 20k val, same mask seed, unified per-base)

`python scripts/compare_all_models.py` → `outputs/compare_all_models.{json,png}`

| model | tokenizer | objective | per-base acc | per-base PPL |
|-------|-----------|-----------|--------------|--------------|
| kmer1-BERT    | kmer1 | MLM    | 75.6% | 1.762 |
| **kmer1-hybrid**  | kmer1 | MLM    | **82.4%** | 1.544 |
| kmer1-decoder | kmer1 | causal | 74.7% | 1.832 |
| kmer6-BERT    | kmer6 | MLM    | 81.3% | 1.518 |
| **kmer6-hybrid**  | kmer6 | MLM    | 82.0% | **1.497** |
| kmer6-decoder | kmer6 | causal | 77.6% | 1.536 |
| random baseline | — | — | 20% | 5.00 |

**Conclusions** (both axes help, and roughly stack):
1. **Backbone (within MLM): hybrid > BERT** for both tokenizers — Transformer+Mamba
   is the better encoder. Largest gain at kmer1 (75.6%→82.4%), small at kmer6.
2. **Tokenizer: kmer6 ≥ kmer1** broadly (non-overlapping k-mer encodes local
   structure). Best PPL = kmer6-hybrid (1.497); best accuracy = kmer1-hybrid (82.4%).
3. **More data helps**: the full-28M kmer6 model beats the earlier 1M-subset
   kmer6 (per-base acc 81.2% vs 78.5%, `outputs/bert_mlm/fm_kmer6_vs_subset.png`),
   despite 1 epoch vs ~27 epochs — diversity > repetition.
4. **Objective caveat**: MLM "perplexity" is a *pseudo*-PPL on 15% masked tokens
   with **bidirectional** context; causal/decoder PPL is true autoregressive over
   **every** token with **left-only** context — intrinsically harder. Compare
   **within** an objective for rigor; the decoders are competitive nonetheless
   (kmer6-decoder 1.536), and show the same tokenizer effect (1.536 vs 1.832).

## Comparison scripts & figures

| script | output | what |
|--------|--------|------|
| `compare_all_models.py` | `compare_all_models.{json,png}` | all 6 models, unified per-base (the master comparison) |
| `compare_fm_backbones.py` | `fm_backbone_compare.{json,png}` | BERT vs hybrid (kmer1 & kmer6) |
| `eval_kmer_perbase.py` | — | per-base acc + PPL for one model/k |
| `plot_fm_live.py` | `bert_mlm/fm_live.png` | live training curves of the 4 MLM runs |
| `plot_decoder_live.py` | `fm_decoder_live.png` | live training curves of the 2 decoder runs |

> Mamba install (for the hybrid backbone) is documented in the
> [Transformer+Mamba hybrid backbone](#transformermamba-hybrid-backbone-kmer1)
> section — use the prebuilt wheels with `--no-deps` (PyPI install upgrades
> torch and breaks the env).
