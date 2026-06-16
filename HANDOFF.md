# HANDOFF — Hybrid-650M vs RiNALMo-Giga (continue on H200)

This repo trains an **RNA Hybrid (Transformer+Mamba) foundation model** and compares it to
**RiNALMo-Giga (650M)** on the **ArchiveII leave-one-family-out (LFO) secondary-structure**
benchmark, under a deliberately RiNALMo-aligned pretraining budget and the **exact official
RiNALMo downstream pipeline**.

You (the new instance on the H200 box) should read this top to bottom, then run the commands in
§6. The L20 box it was started on is too small (8×46 GB); H200 (141 GB) lets you raise the
micro-batch a lot and finish far faster.

---

## 1. The one question we are answering
> Under matched data / tokenizer / context / training budget / downstream protocol, can
> **Hybrid-650M** reach or beat **RiNALMo-Giga** on ArchiveII LFO secondary structure?

Only the **backbone** differs (our hybrid vs their pure Transformer). Everything else is matched.

## 2. Current state (what was done on the L20 box)
- **Model**: `HybridMambaForMaskedLM`, **664.3M params** — hidden 1280, **33 layers `TTM`×11**
  (22 Transformer / 11 Mamba), 20 heads, FFN 5120, mamba_expand 4, ctx 1022 (max_pos 1026), SDPA.
- **Pretraining**: full RNAcentral (40.7M seqs, 30.7B bases), ctx 1022, random-crop >1022, MLM
  15%/80-10-10, AdamW 5e-5 → cosine→1e-5 (warmup 2000), effective batch 1344. Got through
  **~epoch 1 of 6** (MLM loss plateaued ~0.37). Checkpoint: `outputs/fm_hybrid_650m/checkpoint-30278`
  (also on HF, see §5). **Pretraining was NOT finished** — this is the main thing to do on H200.
- **Downstream**: faithful RiNALMo SS pipeline, fully validated (see §3). On the epoch-1 model the
  LFO macro-F1 (8 families, correct lr 5e-4) = **0.476** vs RiNALMo-Giga **0.694** on the same 8.

## 3. KEY FINDINGS — read before re-deriving anything
1. **Our eval pipeline is PROVEN correct.** We ran RiNALMo's *own* released giga 5s fine-tuned
   weights (`zenodo 15043668`) through OUR decoder+metric → **F1 0.882**, reproducing their paper
   **0.88**. And our `prob_mat_to_sec_struct`+`ss_f1` are **bit-identical** to RiNALMo's
   (`rinalmo/utils/sec_struct.py`) on 224 test cases (0 diff). ⇒ the gap is the **model**, not eval.
   - Reproduce: `python scripts/eval_rinalmo_weights.py --family 5s --weights <giga_ss_5s_ft.pt>`
2. **Downstream FT lr was a real bug, now fixed.** RiNALMo official SS-FT uses **lr 5e-4**
   (`train_sec_struct_prediction.py` default), not 1e-5. Fixed in `rinalmo_ss_finetune.py`
   (`--base_lr 5e-4`, default). It only moved test macro +0.014 though — not the main gap.
3. **The gap is pretraining, not FT.** We're at 1/6 epochs. Two faithfulness gaps to close on H200:
   - **(a) finish 6 epochs** (biggest lever).
   - **(b) cluster the pretraining data**: RiNALMo de-duplicates RNAcentral to **~17M reps/epoch**;
     we used the full 40.7M (redundant). Implement clustering (mmseqs/CD-HIT) before the real run —
     see §6 step 0. This is an open faithfulness item.

## 4. Faithfulness checklist (verified vs RiNALMo official code — keep it this way)
| item | status |
|---|---|
| single-nt tokenizer (A U C G N + specials) | ✅ `tokenizers/single` |
| ctx 1022, random-crop >1022 (re-sampled/epoch) | ✅ `rna_stream_dataset.py` |
| MLM 15% / 80-10-10, loss on masked only | ✅ `MLMCollator` |
| SS head = outer_concat + Linear(2d→64) + **2** ResNet2D | ✅ `rinalmo_ss_lib.py` (num_blocks=2) |
| SS loss = upper-tri diag=1 BCE, **no** sharp mask in loss | ✅ |
| canonical/sharp masking only in **decoder** | ✅ |
| strip CLS/EOS before head | ✅ |
| threshold 0.01–0.29 tuned on val avg-F1 | ✅ |
| FT lr 5e-4, Adam, LinearLR 1.0→0.1 / 7000 steps | ✅ |
| gradual unfreeze: 3 layers / 3 epochs top-down, keep bottom 9 frozen (giga YAML) | ✅ `--unfreeze_min_layer 9` |
| pretraining data clustered to ~17M/epoch | ❌ **TODO on H200** |

## 5. Environment (delicate — replicate exactly)
- conda env `rnafm`: torch **2.5.1+cu121**, transformers **4.46.3** (do NOT upgrade — 5.x breaks vocab.txt),
  datasets 3.1.0.
- `mamba-ssm` 2.2.4 + `causal-conv1d` 1.5.0.post8 via **prebuilt wheels** `--no-deps`
  (`cu12torch2.5cxx11abiFALSE`); a normal `pip install mamba-ssm` will upgrade torch and break things.
- `flash-attn` 2.7.4.post1 prebuilt wheel `--no-deps` (only needed to run RiNALMo's model for the
  §3 validation): `pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`
- For the RiNALMo-weights validation: clone `https://github.com/lbcb-sci/RiNALMo` to `/tmp/RiNALMo`,
  `pip install ml_collections`, and patch `rinalmo/model/attention.py` line ~209
  `unpad_input(...)` to unpack with `*_` (flash-attn ≥2.5 returns 5 values).

## 6. RUN ON H200 (commands)

```bash
cd <repo>
source <conda>/etc/profile.d/conda.sh && conda activate rnafm

# 0. (TODO/recommended) cluster RNAcentral to ~17M reps before the real run, e.g.:
#    mmseqs easy-cluster data/raw/rnacentral_active.fasta clust tmp --min-seq-id 0.x ...
#    then point rna_stream_dataset.build_shm_cache at the clustered fasta.

# 1. Build the /dev/shm streaming cache (decompresses RNAcentral to RAM, ~5 min)
python scripts/rna_stream_dataset.py            # writes /dev/shm/rna_seqs.txt (+ val_seqs.txt)

# 2. Pretrain Hybrid-650M. ON H200 RAISE micro_batch (141 GB): try 24-48 and drop grad_accum
#    so micro_batch * n_gpu * grad_accum == 1344. Re-benchmark first (see note below).
bash scripts/run_hybrid650.sh                   # edit --micro_batch/--grad_accum/--nproc_per_node for H200
#    Resumes automatically if outputs/fm_hybrid_650m/checkpoint-* exists (ignore_data_skip set).
#    Live curves: scripts/live_plot_650m.py ; per-epoch SS eval: scripts/epoch_ss_eval_watcher.py

# 3. Per-epoch downstream comparison (full gradual-unfreeze FT, 9 families):
#    point MODEL at an epoch checkpoint dir (needs model_config.json + pytorch_model.bin)
bash scripts/run_ss_ft_650m_ep1.sh              # uses lr 5e-4 + giga unfreeze; edit MODEL/OUT/EPOCHS

# 4. Validate eval pipeline against RiNALMo's own weights (optional, already done = 0.882≈0.88):
#    wget the per-family giga ss weights from zenodo 15043668, then:
python scripts/eval_rinalmo_weights.py --family 5s --weights giga_ss_5s_ft.pt --device cuda:0
```

**H200 batch-size note:** on L20 we were memory-bound at micro_batch 8 (SDPA, no checkpointing).
H200 has 141 GB — micro_batch can go far higher → fewer grad-accum steps → much higher MFU and
~linear speedup. Re-run the throughput micro-benchmark (instantiate the 664M config, time fwd+bwd
at ctx 1022 across micro_batch 16/32/48/64) and pick the largest that fits, keeping
`micro_batch * n_gpu * grad_accum == 1344`.

## 7. Key files
- Model: `models/hybrid_mamba_bert.py` (SDPA default + gradient_checkpointing supported)
- Pretrain: `scripts/train_hybrid650_mlm.py`, `scripts/rna_stream_dataset.py`, `scripts/run_hybrid650.sh`
- SS pipeline (faithful RiNALMo): `scripts/rinalmo_ss_lib.py` (head/decoder/metric), `scripts/rinalmo_ss_finetune.py` (gradual-unfreeze FT), `scripts/run_ss_ft_650m_ep1.sh`
- Per-epoch eval + curves: `scripts/epoch_ss_eval_watcher.py`, `scripts/live_plot_650m.py`
- Pipeline validation: `scripts/eval_rinalmo_weights.py`
- Stats: `scripts/full_rnacentral_stats.py` → `outputs/fm_hybrid_650m/full_rnacentral_stats.json`

## 8. Results so far (epoch-1 model, lr 5e-4 FT, flexible F1)
| family | Hybrid-650M ep1 | RiNALMo-Giga |
|---|---|---|
| 5s | 0.785 | 0.88 |
| 16s | 0.552 | 0.74 |
| 23s | 0.628 | 0.85 |
| grp1 | 0.139 | 0.66 |
| srp | 0.413 | 0.70 |
| telomerase | 0.041 | 0.12 |
| RNaseP | 0.598 | 0.80 |
| tmRNA | 0.648 | 0.80 |
| tRNA | (rerun pending) | 0.93 |
| **macro (8)** | **0.476** | **0.694** |

Expectation: closing (a)+(b) in §3 should move the macro substantially toward RiNALMo. The eval
pipeline is not the bottleneck (proven in §3).
