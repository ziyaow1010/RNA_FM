#!/usr/bin/env python3
"""Aggregate leave-one-family-out supervised contact-prediction results into the
formal RiNALMo-style table: per (held-out family, model) metrics + per-model
macro average over families. Writes compare_archiveII_lfo.{csv,json,png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CP = PROJECT_ROOT / "outputs" / "contact_pred"

MODELS = ["kmer1-BERT", "kmer1-Hybrid", "kmer6-BERT", "kmer6-Hybrid"]
METRICS = ["precision", "recall", "F1", "best_F1", "MCC", "AUPRC", "P_at_L", "P_at_L2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", required=True)
    p.add_argument("--out_prefix", default=str(CP / "compare_archiveII_lfo"))
    args = p.parse_args()

    rows, nested = [], {}
    for model in MODELS:
        nested[model] = {}
        for fam in args.families:
            tm = CP / model / f"archiveII_lfo_{fam}" / "test_metrics.json"
            if not tm.exists():
                continue
            sm = json.load(open(tm))
            rec = {"heldout_family": fam, "model": model,
                   "num_test": sm.get("num_test")}
            for m in METRICS:
                rec[f"mean_{m}"] = sm.get(f"mean_{m}")
            rows.append(rec)
            nested[model][fam] = {m: sm.get(f"mean_{m}") for m in METRICS}

    # macro average per model (mean over families)
    macro = {}
    for model in MODELS:
        fam_vals = nested[model]
        if not fam_vals:
            continue
        mac = {"heldout_family": "MACRO_AVG", "model": model,
               "num_test": sum(1 for _ in fam_vals)}
        for m in METRICS:
            vals = [fam_vals[f][m] for f in fam_vals if fam_vals[f][m] == fam_vals[f][m]]
            mac[f"mean_{m}"] = round(statistics.mean(vals), 4) if vals else float("nan")
        rows.append(mac)
        macro[model] = mac

    fields = ["heldout_family", "model", "num_test"] + [f"mean_{m}" for m in METRICS]
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    json.dump({"per_family": [r for r in rows if r["heldout_family"] != "MACRO_AVG"],
               "macro_avg": macro}, open(args.out_prefix + ".json", "w"), indent=2)

    # ---- figure: (left) macro-avg bars; (right) best_F1 heatmap model x family ----
    fig = plt.figure(figsize=(17, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    keys = [("mean_F1", "F1"), ("mean_best_F1", "best_F1"), ("mean_MCC", "MCC"), ("mean_AUPRC", "AUPRC")]
    x = np.arange(len(MODELS)); w = 0.2
    for i, (k, lab) in enumerate(keys):
        vals = [macro.get(mdl, {}).get(k, 0) or 0 for mdl in MODELS]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=lab)
    ax1.set_xticks(x); ax1.set_xticklabels(MODELS, rotation=15, fontsize=9)
    ax1.set_title("Macro-average over held-out families"); ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    grid = np.full((len(MODELS), len(args.families)), np.nan)
    for mi, model in enumerate(MODELS):
        for fi, fam in enumerate(args.families):
            v = nested[model].get(fam, {}).get("best_F1")
            if v is not None:
                grid[mi, fi] = v
    im = ax2.imshow(grid, cmap="viridis", aspect="auto", vmin=0)
    ax2.set_xticks(range(len(args.families))); ax2.set_xticklabels(args.families, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(MODELS))); ax2.set_yticklabels(MODELS, fontsize=9)
    for mi in range(len(MODELS)):
        for fi in range(len(args.families)):
            if grid[mi, fi] == grid[mi, fi]:
                ax2.text(fi, mi, f"{grid[mi,fi]:.2f}", ha="center", va="center",
                         color="white" if grid[mi, fi] < grid[np.isfinite(grid)].max()*0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046); ax2.set_title("best-F1 per held-out family")
    fig.suptitle("RiNALMo-style ArchiveII leave-one-family-out — supervised contact prediction "
                 "(frozen LM + ResNet head)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=110)

    print(f"[lfo-agg] wrote {args.out_prefix}.{{csv,json,png}}")
    print(f"\n{'model':14}{'F1':>8}{'best_F1':>9}{'MCC':>8}{'AUPRC':>8}{'P@L':>8}  (macro avg over {len(args.families)} families)")
    for model in MODELS:
        m = macro.get(model)
        if m:
            print(f"{model:14}{m['mean_F1']:>8.3f}{m['mean_best_F1']:>9.3f}{m['mean_MCC']:>8.3f}"
                  f"{m['mean_AUPRC']:>8.3f}{m['mean_P_at_L']:>8.3f}")


if __name__ == "__main__":
    main()
