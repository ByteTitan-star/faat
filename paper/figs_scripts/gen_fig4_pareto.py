#!/usr/bin/env python3
"""F4 — Stealth-vs-effectiveness Pareto on CIFAR-10.
x = SSIM (higher = stealthier), y = ASR (higher = stronger).
FAAT: v4 L2 sweep {0.9,1.2,1.5} (mean of 3 seeds) connected; the L2=1.5 point is the operating point.
Narcissus & BppAttack: single points from the saturation/Pareto study (docs/SATURATION-ANALYSIS.md).
Source: data_faat_summary.csv (cifar10_v4 rows).
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save, color

setup()
rows = list(csv.DictReader(open("_data/data_faat_summary.csv")))
faat = [(float(r["SSIM_mean"]), float(r["ASR_mean"]), float(r["L2"]))
        for r in rows if r["file_tag"] == "cifar10_v4" and float(r["L2"]) <= 1.5]
faat.sort()

fig, ax = plt.subplots(figsize=(3.5, 2.7))
xs = [p[0] for p in faat]; ys = [p[1] for p in faat]
ax.plot(xs, ys, "-o", color=color("FAAT"), lw=1.6, ms=5, zorder=4, label="FAAT (Ours)")
for s, a, l2 in faat:
    ax.annotate(f"L2={l2}", (s, a), textcoords="offset points",
                xytext=(4, -9), fontsize=6, color=color("FAAT"))

# reference points (ASR/SSIM) from docs/SATURATION-ANALYSIS.md (Pareto front)
refs = [
    ("Narcissus",   0.88, 0.946, color("Narcissus"), "s"),
    ("BppAttack",   0.95, 0.972, color("BppAttack"), "D"),
    ("BadNets-C",   0.706, 0.90,  color("BadNets"),  "v"),
    ("Blended-C",   0.759, 0.90,  color("Blended"),  "v"),
]
for name, asr, ssim, c, m in refs:
    ax.scatter([ssim], [asr], s=28, marker=m, color=c, edgecolor="black",
               linewidth=0.4, zorder=3, label=name)

ax.set_xlabel("Stealth (SSIM $\\uparrow$)")
ax.set_ylabel("Attack Success Rate (\\%)")
ax.set_xlim(0.89, 1.005); ax.set_ylim(55, 100)
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.set_axisbelow(True)
save(fig, "f4_pareto")
