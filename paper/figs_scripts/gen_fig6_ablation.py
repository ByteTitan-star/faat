#!/usr/bin/env python3
"""F6 — Component ablation on CIFAR-10 (L2=1.5).
Shows the role of L_align (stealth, not ASR): removing it raises ASR slightly
but worsens defense-evasion (SS-AUC drops toward 0). Full FAAT is the operating point.
Sources: data_ablation.csv (w/o align), data_faat_summary.csv (Full), and the
v3.1 adaptive-ablation numbers from docs/FAAT-EXPERIMENTS.md for the w/o adaptive bar.
"""
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save, color

setup()
# variant: (ASR, SS_AUC, SSIM)  — ASR & SSIM higher better; SS_AUC closer to 0.5 = more evasive
variants = [
    ("$\\delta_g$ only\n(Narcissus)", 90.3, 0.385, 0.963),  # v3.1 scale0.2 no-adaptive (FAAT-EXPERIMENTS.md)
    ("w/o $\\delta_a$",        90.3, 0.385, 0.963),
    ("w/o $\\mathcal{L}_{align}$", 96.4, 0.374, 0.950),       # data_ablation.csv (seed1)
    ("Full FAAT",              93.6, 0.433, 0.950),          # data_faat_summary.csv
]
# dedupe the first two (same numbers) -> relabel
variants = [
    ("$\\delta_g$ only",   90.3, 0.385, 0.963),
    ("w/o $\\mathcal{L}_{align}$", 96.4, 0.374, 0.950),
    ("Full FAAT",          93.6, 0.433, 0.950),
]
names = [v[0] for v in variants]
asr = [v[1] for v in variants]
ssauc = [v[2] for v in variants]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
xx = np.arange(len(names))
cols = ["#bbbbbb", "#8e7cc3", color("FAAT")]
ax1.bar(xx, asr, color=cols, edgecolor="black", linewidth=0.5)
for i, v in enumerate(asr): ax1.text(i, v+0.6, f"{v:.1f}", ha="center", fontsize=6.5)
ax1.set_xticks(xx); ax1.set_xticklabels(names, fontsize=6.5)
ax1.set_ylabel("ASR (\\%)"); ax1.set_ylim(85, 100); ax1.set_title("(a) Effectiveness", fontsize=8)

ax2.bar(xx, ssauc, color=cols, edgecolor="black", linewidth=0.5)
for i, v in enumerate(ssauc): ax2.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=6.5)
ax2.axhline(0.5, color="#aaa", lw=0.7, ls="--"); ax2.text(len(names)-0.4, 0.505, "undetectable", fontsize=5.8, color="#777", ha="right")
ax2.set_xticks(xx); ax2.set_xticklabels(names, fontsize=6.5)
ax2.set_ylabel("SS-AUC (detection)"); ax2.set_ylim(0.30, 0.55)
ax2.set_title("(b) Defense evasion (SS)", fontsize=8)
ax2.set_axisbelow(True)
save(fig, "f6_ablation")
