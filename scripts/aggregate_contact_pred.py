#!/usr/bin/env python3
"""Aggregate supervised contact-prediction test_metrics.json across the 4 FMs
into one comparison table + barplot (Part 8). FM backbone params are read from
outputs/compare_all_models.json when available."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "contact_pred"

# model_name -> (tokenizer, backbone, key in compare_all_models.json)
META = {
    "kmer1-BERT":   ("kmer1", "BERT",   "kmer1-BERT"),
    "kmer1-Hybrid": ("kmer1", "Hybrid", "kmer1-hybrid"),
    "kmer6-BERT":   ("kmer6", "BERT",   "kmer6-BERT"),
    "kmer6-Hybrid": ("kmer6", "Hybrid", "kmer6-hybrid"),
}
COLUMNS = ["model", "tokenizer", "backbone", "params", "split",
           "num_train", "num_val", "num_test",
           "mean_precision", "mean_recall", "mean_F1", "mean_best_F1",
           "mean_MCC", "mean_AUPRC", "mean_P_at_L", "mean_P_at_L2"]


def fm_params():
    f = PROJECT_ROOT / "outputs" / "compare_all_models.json"
    if f.exists():
        d = json.load(open(f))
        return {k: v.get("param_count") for k, v in d.items()}
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entries", nargs="+", required=True, help="model_name=output_dir")
    p.add_argument("--split", default="archiveII_random")
    p.add_argument("--out_prefix", default=str(OUT / "compare_archiveII_random"))
    args = p.parse_args()

    fmp = fm_params()
    rows = []
    for ent in args.entries:
        name, d = ent.split("=", 1)
        sm = json.load(open(Path(d) / "test_metrics.json"))
        tok, bb, key = META.get(name, (name, "?", name))
        rows.append({
            "model": name, "tokenizer": tok, "backbone": bb,
            "params": fmp.get(key), "split": args.split,
            "num_train": sm.get("num_train"), "num_val": sm.get("num_val"),
            "num_test": sm.get("num_test"),
            "mean_precision": sm.get("mean_precision"), "mean_recall": sm.get("mean_recall"),
            "mean_F1": sm.get("mean_F1"), "mean_best_F1": sm.get("mean_best_F1"),
            "mean_MCC": sm.get("mean_MCC"), "mean_AUPRC": sm.get("mean_AUPRC"),
            "mean_P_at_L": sm.get("mean_P_at_L"), "mean_P_at_L2": sm.get("mean_P_at_L2"),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(args.out_prefix + ".json", "w"), indent=2)

    labels = [r["model"] for r in rows]
    metrics = [("mean_F1", "mean F1"), ("mean_best_F1", "mean best-F1"),
               ("mean_MCC", "mean MCC"), ("mean_AUPRC", "mean AUPRC")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:orange"]
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] or 0 for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=colors[:len(rows)])
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Supervised RNA secondary-structure prediction (frozen LM + ResNet head) — "
                 + args.split, fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=120)
    print(f"[aggregate] wrote {args.out_prefix}.{{csv,json,png}}")
    for r in rows:
        print(f"  {r['model']:14} F1={r['mean_F1']:.3f} best_F1={r['mean_best_F1']:.3f} "
              f"MCC={r['mean_MCC']:.3f} AUPRC={r['mean_AUPRC']:.3f} P@L={r['mean_P_at_L']:.3f}")


if __name__ == "__main__":
    main()
