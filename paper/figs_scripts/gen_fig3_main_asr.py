#!/usr/bin/env python3
"""F3 — Main result: ASR across 4 datasets, FAAT vs reproduced/paper baselines.
All at the SAME poison rate FAAT was evaluated at (rate annotated under each group).
Sources: data_baseline_best.csv (CIFAR-10), docs/PAPER-BASELINES.md,
docs/GTSRB-BASELINES.md (cross-dataset baselines); data_faat_summary.csv (FAAT).
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save, color, DASH

setup()
DAT = defaultdict(dict)  # _data not needed; values are aggregated & cited below.

# ---- comparison values (ASR %), with provenance ----
# CIFAR-10 @1%: reproduced (data_baseline_best.csv)
# CIFAR-100 @0.5%, Tiny @0.25%: paper Table values (docs/PAPER-BASELINES.md)
# FAAT: data_faat_summary.csv best config per dataset
# (GTSRB is analyzed separately as a high-confidence stress test, Sec. ref=gtsrb)
methods = ["BadNets-C", "Blended-C", "MultiBpp-RGB", "MultiBpp-B", "FAAT"]
data = {
    # dataset: [BadNets, Blended, MBpp-RGB, MBpp-B, FAAT], rate
    "CIFAR-10\n(1%)":      [70.62, 75.86, 85.09, 89.70, 93.6],
    "CIFAR-100\n(0.5%)":   [85.06, 77.45, None,  None,  98.5],
    "Tiny-IN\n(0.25%)":    [38.96, 43.93, None,  None,  95.85],
}

datasets = list(data.keys())
x = np.arange(len(datasets))
n = len(methods)
w = 0.78 / n
fig, ax = plt.subplots(figsize=(7.0, 2.7))
for i, m in enumerate(methods):
    vals = [data[d][i] for d in datasets]
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + (i - n / 2 + 0.5) * w, plot_vals, w,
                  color=color(m),
                  edgecolor="black" if m == "FAAT" else "none",
                  linewidth=0.8 if m == "FAAT" else 0,
                  hatch="//" if m == "FAAT" else None,
                  label=m if m != "FAAT" else "FAAT (Ours)", zorder=3)
    for j, v in enumerate(vals):
        if v is None:
            ax.text(x[j] + (i - n / 2 + 0.5) * w, 2.0, DASH,
                    ha="center", va="bottom", fontsize=6, color="#888")
        else:
            ax.text(x[j] + (i - n / 2 + 0.5) * w, v + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=5.6,
                    fontweight="bold" if m == "FAAT" else "normal")
ax.set_ylabel("Attack Success Rate (\\%)")
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylim(0, 112)
ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.13),
          frameon=False, columnspacing=1.0, handletextpad=0.4)
ax.set_axisbelow(True)
save(fig, "f3_main_asr")
