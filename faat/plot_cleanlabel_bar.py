"""Bar chart: KST (clean-label 1% 300ep) vs baseline Table 1, by selection strategy."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")
s = json.load(open(os.path.join(OUT, "cleanlabel_summary.json")))
baseline = s["baseline"]
kst = {r["selection"]: r for r in s["kst"]}

selections = ["random", "forget", "res/linear"]
attacks = ["Badnets-C", "Blended-C", "MultiBpp-RGB", "MultiBpp-B"]
col = {"Badnets-C": "#8c564b", "Blended-C": "#e377c2", "MultiBpp-RGB": "#17becf", "MultiBpp-B": "#7f7f7f"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, sel in zip(axes, selections):
    vals = [baseline[sel][a] for a in attacks]
    x = np.arange(len(attacks))
    bars = ax.bar(x, vals, 0.6, color=[col[a] for a in attacks], edgecolor="black", linewidth=0.5, label="baseline")
    # KST line (best across eps for this selection)
    kst_vals = [r["ASR"] for r in s["kst"] if r["selection"] == sel]
    kst_eps = [r["name"].split()[0] + " " + r["name"].split()[1] for r in s["kst"] if r["selection"] == sel]
    for kv, ke in zip(kst_vals, kst_eps):
        ax.axhline(kv, color="#2ca02c" if "ε16" in ke else "#1f77b4", ls="--", lw=1.5,
                   label=f"KST {ke} = {kv:.1f}")
    ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"selection = {sel}", fontsize=10)
    ax.set_ylabel("ASR末20均 (%)" if sel == "random" else "")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle("KST (clean-label, 1% poison, 300ep) vs baseline Table 1 — CIFAR-10\n"
             "(KST flat-spectrum δ_peak=1.0 vs baseline triggers all spectrally structured)",
             fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
figpath = os.path.join(OUT, "cleanlabel_vs_baseline.png")
plt.savefig(figpath, dpi=140, bbox_inches="tight")
print("saved", figpath)
