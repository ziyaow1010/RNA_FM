#!/usr/bin/env python3
"""Hybrid-300M + RiNALMo SS pipeline with RiNALMo's GRADUAL-UNFREEZE fine-tuning.

Faithful to RiNALMo's training: epoch 0 trains only the head (LM frozen); then
every 3 epochs the next-deepest 3 LM layers are unfrozen (top-down) and added to
the optimizer as a new param group at base_lr/10. Base Adam lr 1e-5, LinearLR
1.0->0.1 over 7000 steps. Loss = BCE on the upper triangle (no pos_weight).
Threshold tuned on val (0.01..0.29, flexible F1); test uses RiNALMo's
canonical+greedy decoder and flexible(+/-1) F1. ArchiveII fam-fold split.
Only the backbone differs from RiNALMo (our hybrid-300M instead of RiNALMo LM).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import contact_common as cc          # noqa: E402
from rinalmo_ss_lib import (SecStructPredictionHead, parse_ct,  # noqa: E402
                            prob_mat_to_sec_struct, ss_f1, ss_precision, ss_recall)

THRESHOLDS = [i / 100 for i in range(1, 30)]
LR_DECAY_STEPS = 7000        # RiNALMo LinearLR total_iters


def load_split(ct_dir, max_len):
    items, skipped = [], 0
    for ct in sorted(Path(ct_dir).glob("*.ct")):
        try:
            seq, pair = parse_ct(ct)
        except Exception:
            continue
        if 0 < len(seq) <= max_len:
            items.append({"id": ct.stem, "seq": seq, "pair": pair})
        else:
            skipped += 1
    return items, skipped


def backbone_hidden(model, ids, am):
    """Final hidden states of the hybrid encoder (skip MLM head). [B, T, d]."""
    h = model.embeddings(input_ids=ids, token_type_ids=None)
    ext = (1.0 - am[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
    for layer, is_m in zip(model.layers, model.is_mamba):
        h = layer(h) if is_m else layer(h, attention_mask=ext)[0]
    return h


def encode(tok, seq, device):
    ids = [tok.cls_token_id] + [tok.convert_tokens_to_ids(c) for c in seq] + [tok.sep_token_id]
    t = torch.tensor([ids], device=device)
    return t, torch.ones_like(t)


def logits_for(model, head, tok, seq, device):
    ids, am = encode(tok, seq, device)
    h = backbone_hidden(model, ids, am)[:, 1:1 + len(seq)]   # strip CLS/SEP -> [1,L,d]
    return head(h)[0]                                        # [L,L]


@torch.no_grad()
def eval_tune(model, head, tok, items, device):
    model.eval(); head.eval()
    per_t = {t: [] for t in THRESHOLDS}
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device)).float().cpu().numpy()
        for t in THRESHOLDS:
            per_t[t].append(ss_f1(it["pair"], prob_mat_to_sec_struct(probs, it["seq"], threshold=t)))
    mean = {t: statistics.mean(v) if v else 0.0 for t, v in per_t.items()}
    bt = max(mean, key=mean.get)
    return bt, mean[bt]


@torch.no_grad()
def eval_test(model, head, tok, items, device, thr):
    model.eval(); head.eval()
    P, R, F = [], [], []
    for it in items:
        probs = torch.sigmoid(logits_for(model, head, tok, it["seq"], device)).float().cpu().numpy()
        ss = prob_mat_to_sec_struct(probs, it["seq"], threshold=thr)
        P.append(ss_precision(it["pair"], ss)); R.append(ss_recall(it["pair"], ss)); F.append(ss_f1(it["pair"], ss))
    return P, R, F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--model_dir", default="outputs/fm_hybrid_mamba_kmer1_300m")
    p.add_argument("--ct_root", default="data/contact_eval/raw/ct/fam-fold")
    p.add_argument("--out_dir", default="outputs/contact_pred/rinalmo_ss_ft")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--base_lr", type=float, default=5e-4)     # RiNALMo official SS-FT lr
    p.add_argument("--unfreeze_block", type=int, default=3)   # layers unfrozen per step
    p.add_argument("--unfreeze_every", type=int, default=3)   # epochs between unfreezes
    p.add_argument("--unfreeze_min_layer", type=int, default=0,
                   help="stop unfreezing at this layer idx (keep [0..min-1] frozen). giga keeps bottom 9.")
    p.add_argument("--tune_every", type=int, default=5)
    p.add_argument("--max_train", type=int, default=0, help="cap train items (0=all); for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = cc.load_model(args.model_dir, "hybrid", device)
    tok = cc.load_tokenizer("tokenizers/single")
    nlayers = len(model.layers)
    head = SecStructPredictionHead(model.config.hidden_size, num_blocks=2).to(device)

    root = Path(args.ct_root) / args.family
    train, sk_tr = load_split(root / "train", args.max_len)
    val, sk_va = load_split(root / "valid", args.max_len)
    test, sk_te = load_split(root / "test", args.max_len)
    if args.max_train:
        train = train[:args.max_train]
    print(f"[{args.family}] train/val/test = {len(train)}/{len(val)}/{len(test)} "
          f"(skipped>{args.max_len}nt: {sk_tr}/{sk_va}/{sk_te})  layers={nlayers}")

    # freeze backbone, head trainable
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in head.parameters():
        prm.requires_grad_(True)

    opt = torch.optim.Adam([{"params": list(head.parameters())}], lr=args.base_lr)
    base_lrs = [args.base_lr]

    # unfreeze schedule: epoch -> [layer indices], top-down 3 at a time,
    # stopping at unfreeze_min_layer (RiNALMo giga keeps the bottom 9 frozen).
    unfreeze_at = {}
    top = nlayers
    e = args.unfreeze_every
    while top > args.unfreeze_min_layer:
        lo = max(args.unfreeze_min_layer, top - args.unfreeze_block)
        unfreeze_at[e] = list(range(top - 1, lo - 1, -1))
        top = lo; e += args.unfreeze_every

    lossfn = nn.BCEWithLogitsLoss()
    gstep = 0
    unfrozen = set()                                  # layer indices currently trainable
    best_val_f1, best_thr, best_head, best_lm = -1.0, 0.5, None, None
    for ep in range(args.epochs):
        if ep in unfreeze_at:
            new_params = []
            for li in unfreeze_at[ep]:
                unfrozen.add(li)
                for prm in model.layers[li].parameters():
                    prm.requires_grad_(True); new_params.append(prm)
            opt.add_param_group({"params": new_params, "lr": args.base_lr / 10.0})
            base_lrs.append(args.base_lr / 10.0)
            print(f"  [{args.family}] epoch {ep}: unfroze layers {unfreeze_at[ep]} (lr {args.base_lr/10:.1e})")

        # frozen backbone stays in eval mode (dropout off, as RiNALMo's freeze());
        # only the head + already-unfrozen layers run in train mode.
        model.eval(); head.train()
        for li in unfrozen:
            model.layers[li].train()
        order = np.random.permutation(len(train))
        tot = 0.0
        t0 = time.perf_counter()
        for k in order:
            it = train[k]; L = len(it["seq"])
            logits = logits_for(model, head, tok, it["seq"], device)
            ut = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            target = torch.from_numpy(it["pair"]).to(device)
            loss = lossfn(logits[ut], target[ut])
            opt.zero_grad(); loss.backward(); opt.step()
            gstep += 1
            factor = 1.0 - 0.9 * min(gstep / LR_DECAY_STEPS, 1.0)
            for gi, grp in enumerate(opt.param_groups):
                grp["lr"] = base_lrs[gi] * factor
            tot += loss.item()
        dt = time.perf_counter() - t0
        print(f"  [{args.family}] epoch {ep} train {len(train)} items in {dt:.0f}s "
              f"({len(train)/max(dt,1e-9):.1f} it/s, {len(unfrozen)} layers unfrozen) loss={tot/len(train):.4f}", flush=True)
        if (ep + 1) % args.tune_every == 0 or ep == args.epochs - 1:
            thr, vf1 = eval_tune(model, head, tok, val, device)
            print(f"  [{args.family}] epoch {ep+1} loss={tot/len(train):.4f} val_F1={vf1:.4f} thr={thr}", flush=True)
            if vf1 > best_val_f1:
                best_val_f1, best_thr = vf1, thr
                best_head = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                best_lm = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_head is not None:
        head.load_state_dict(best_head); model.load_state_dict(best_lm)
    P, R, F = eval_test(model, head, tok, test, device, best_thr)
    res = {"family": args.family, "backbone": "hybrid-300M", "pipeline": "RiNALMo+gradual-unfreeze-FT",
           "num_train": len(train), "num_val": len(val), "num_test": len(test),
           "tuned_threshold": best_thr, "best_val_F1": best_val_f1,
           "mean_F1": statistics.mean(F) if F else 0.0,
           "mean_precision": statistics.mean(P) if P else 0.0,
           "mean_recall": statistics.mean(R) if R else 0.0,
           "median_F1": statistics.median(F) if F else 0.0, "metric": "flexible(+/-1)_F1"}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{args.family}.json", "w"), indent=2)
    print(f"[{args.family}] TEST F1={res['mean_F1']:.4f} P={res['mean_precision']:.4f} "
          f"R={res['mean_recall']:.4f} thr={best_thr} n={len(test)}", flush=True)


if __name__ == "__main__":
    main()
