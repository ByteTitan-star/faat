"""Pareto + defense figure for KST epsilon sweep vs Narcissus on CIFAR-10."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")

def load_res(tag):
    f = os.path.join(OUT, f"{tag}_result.json")
    return json.load(open(f)) if os.path.exists(f) else None

# KST sweep
kst = []
for e in [8, 12, 16, 20]:
    eps = f"{e/255.0:.3f}"
    r = load_res(f"kst_eps{eps}_pr0.05_s1")
    if r: kst.append((e, r))
narc = load_res("narcissus_eps0.031_pr0.05_s1")

# defense eval
defs = {}
df = os.path.join(OUT, "defense_eval.json")
if os.path.exists(df):
    for row in json.load(open(df)):
        defs[row["tag"]] = row

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# (1) ASR vs SSIM Pareto
ax = axes[0]
kssim = [r["SSIM"] for _, r in kst]
kasr = [r["ASR"] for _, r in kst]
ax.plot(kssim, kasr, "o-", color="#2ca02c", lw=2, ms=9, label="KST (ε sweep)")
for e, r in kst:
    ax.annotate(f"ε={e}/255", (r["SSIM"], r["ASR"]), textcoords="offset points",
                xytext=(5, -12), fontsize=7)
if narc:
    ax.scatter([narc["SSIM"]], [narc["ASR"]], color="#d62728", s=120, marker="*",
               zorder=5, label="Narcissus (ε=8/255)")
ax.set_xlabel("SSIM (stealth ↑)"); ax.set_ylabel("ASR (attack ↑)")
ax.set_title("Pareto: ASR vs stealth", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(0.94, 1.0)

# (2) spectral peak (log)
ax = axes[1]
labels = [f"KST\nε={e}/255" for e, _ in kst] + ["Narcissus\nε=8/255"]
peaks = [r["spectral_peak_over_median"] for _, r in kst] + ([narc["spectral_peak_over_median"]] if narc else [])
colors = ["#2ca02c"]*len(kst) + (["#d62728"] if narc else [])
ax.bar(range(len(labels)), peaks, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yscale("log"); ax.set_ylabel("max|FFT(δ)|² / median")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
ax.set_title("(a) spectral flatness of trigger", fontsize=10)
ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="flat (=1)")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(peaks):
    ax.text(i, v*1.3, f"{v:.1f}", ha="center", fontsize=7)

# (3) defence: STRIP AUC (left) + frequency-signature set-level peak (right, log)
ax = axes[2]
tags = [f"kst_eps{e/255.0:.3f}_pr0.05_s1" for e, _ in kst]
strip_k = [defs.get(t, {}).get("STRIP_AUC", np.nan) for t in tags]
freqpk_k = [defs.get(t, {}).get("FreqSig_peak_over_median", np.nan) for t in tags]
narc_tag = "narcissus_eps0.031_pr0.05_s1"
strip_n = defs.get(narc_tag, {}).get("STRIP_AUC", np.nan)
freqpk_n = defs.get(narc_tag, {}).get("FreqSig_peak_over_median", np.nan)
x = np.arange(len(kst) + 1)                      # KST sweep points + Narcissus
w = 0.38
strip_vals = strip_k + [strip_n]
freqpk_vals = freqpk_k + [freqpk_n]
ax.bar(x - w/2, strip_vals, w, color="#ff7f0e", edgecolor="black", linewidth=0.5, label="STRIP AUC (behaviour)")
ax.set_ylim(0, 1.05); ax.set_ylabel("STRIP AUC (↑=defence)")
ax.axhline(0.5, color="gray", ls="-", lw=0.8, alpha=0.5)
ax2 = ax.twinx()
ax2.bar(x + w/2, freqpk_vals, w, color="#2ca02c", edgecolor="black", linewidth=0.5, label="freq-signature peak/med")
ax2.set_yscale("log"); ax2.set_ylim(1, 2e4); ax2.set_ylabel("freq-signature peak/med (↑=defence)")
ax.set_xticks(x); ax.set_xticklabels([f"ε={e}/255" for e, _ in kst] + ["Narc\nε=8"], fontsize=8)
ax.set_title("defence evasion  (low = attack evades)", fontsize=10)
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper left"); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
figpath = os.path.join(OUT, "kst_pareto_defense.png")
plt.savefig(figpath, dpi=140, bbox_inches="tight")
print("saved", figpath)
print("KST:", [(e, r["ASR"], r["SSIM"], r["spectral_peak_over_median"]) for e, r in kst])
print("Narc:", narc)
print("defs:", defs)
