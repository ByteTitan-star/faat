#!/usr/bin/env python3
"""F5 — Defense robustness: post-defense ASR for res_square vs random selection.
Shows the Component-A (res_square) selection is consistently harder to defend.
Source: data_defense.csv (extracted from the baseline authors' defense logs —
standard attacks badnet/blend/sig/ctrl; NOT a FAAT-vs-BackdoorBench claim).
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save, color

setup()
rows = list(csv.DictReader(open("_data/data_defense.csv")))
defs = ["ac", "nc", "fst", "fp", "abl", "rnp", "i-bau"]
pretty = {"ac":"AC","nc":"NC","fst":"FST","fp":"FP","abl":"ABL","rnp":"RNP","i-bau":"I-BAU"}
val = defaultdict(dict)
for r in rows:
    if r["defense"] in defs and r["test_asr_mean"] != "":
        val[r["selection"]][r["defense"]] = float(r["test_asr_mean"])

x = np.arange(len(defs)); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 2.5))
rs = [val["res_square"].get(d, 0) for d in defs]
rd = [val["random"].get(d, 0) for d in defs]
ax.bar(x - w/2, rd, w, color=color("random"), edgecolor="#999", label="random selection", zorder=3)
ax.bar(x + w/2, rs, w, color=color("res_square"), label="Res-x$^2$ selection (ours)", zorder=3)
for i, (a, b) in enumerate(zip(rd, rs)):
    ax.text(i - w/2, a + 1.2, f"{a:.0f}", ha="center", fontsize=5.8)
    ax.text(i + w/2, b + 1.2, f"{b:.0f}", ha="center", fontsize=5.8, fontweight="bold")
ax.axhline(50, color="#aaa", lw=0.7, ls="--", zorder=1)
ax.text(len(defs)-0.4, 51, "chance", fontsize=5.5, color="#999", ha="right")
ax.set_xticks(x); ax.set_xticklabels([pretty[d] for d in defs])
ax.set_ylabel("Post-defense ASR (\\%)")
ax.set_ylim(0, 90)
ax.legend(frameon=False, loc="upper right", fontsize=6.5)
ax.set_axisbelow(True)
save(fig, "f5_defense")
