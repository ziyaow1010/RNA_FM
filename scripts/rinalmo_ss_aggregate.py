#!/usr/bin/env python3
"""Aggregate Hybrid-300M + RiNALMo-pipeline LFO results: family + macro F1,
and compare to our own contact-head pipeline (same backbone, strict metric)."""
import json, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CP = Path("outputs/contact_pred")
fams = "tRNA 5s RNaseP srp tmRNA 16s 23s grp1 telomerase".split()
rows = {}
for f in fams:
    p = CP / "rinalmo_ss" / f"{f}.json"
    if p.exists(): rows[f] = json.load(open(p))
# our own pipeline (kmer1-Hybrid-300M, strict metric, our head) best_F1 for ref
def ours(f):
    p = CP / "kmer1-Hybrid-300M" / f"archiveII_lfo_{f}" / "test_metrics.json"
    return json.load(open(p)).get("mean_best_F1") if p.exists() else None
macro_f1 = statistics.mean([rows[f]["mean_F1"] for f in rows])
macro_p = statistics.mean([rows[f]["mean_precision"] for f in rows])
macro_r = statistics.mean([rows[f]["mean_recall"] for f in rows])
out = {"backbone": "hybrid-300M", "pipeline": "RiNALMo (flexible F1, canonical+greedy decoder)",
       "macro_F1": round(macro_f1,4), "macro_precision": round(macro_p,4), "macro_recall": round(macro_r,4),
       "per_family": {f: {"F1": round(rows[f]["mean_F1"],4), "precision": round(rows[f]["mean_precision"],4),
                          "recall": round(rows[f]["mean_recall"],4), "threshold": rows[f]["tuned_threshold"],
                          "n_test": rows[f]["num_test"]} for f in rows}}
json.dump(out, open(CP / "compare_rinalmo_ss_lfo.json", "w"), indent=2)
print(f"\n{'family':12}{'F1':>8}{'prec':>8}{'rec':>8}{'thr':>6}{'n_test':>8}   (ours best_F1)")
for f in fams:
    if f in rows:
        r=rows[f]; o=ours(f)
        print(f"{f:12}{r['mean_F1']:>8.3f}{r['mean_precision']:>8.3f}{r['mean_recall']:>8.3f}"
              f"{r['tuned_threshold']:>6}{r['num_test']:>8}   ({o if o else '-'})")
print(f"\nMACRO  F1={macro_f1:.4f}  P={macro_p:.4f}  R={macro_r:.4f}")
# plot
fig,ax=plt.subplots(figsize=(11,5)); fs=[f for f in fams if f in rows]
ax.bar(range(len(fs)), [rows[f]['mean_F1'] for f in fs], color='tab:orange')
for i,f in enumerate(fs): ax.text(i, rows[f]['mean_F1'], f"{rows[f]['mean_F1']:.2f}", ha='center', va='bottom')
ax.set_xticks(range(len(fs))); ax.set_xticklabels(fs, rotation=30); ax.set_ylabel('flexible F1'); ax.set_ylim(0,1)
ax.axhline(macro_f1, ls='--', color='gray'); ax.text(0, macro_f1+0.01, f'macro={macro_f1:.3f}', color='gray')
ax.set_title('Hybrid-300M + RiNALMo SS pipeline — ArchiveII leave-one-family-out (flexible F1)')
fig.tight_layout(); fig.savefig(CP/'compare_rinalmo_ss_lfo.png', dpi=120)
print(f"wrote {CP}/compare_rinalmo_ss_lfo.{{json,png}}")
