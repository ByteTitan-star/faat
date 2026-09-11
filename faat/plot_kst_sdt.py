"""Quick comparison figure for KST/SDT/Narcissus CIFAR-10 validation."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")

def load(t):
    f = os.path.join(OUT, f"{t}_eps0.031_pr0.05_s1_result.json")
    return json.load(open(f))

K, S, N = load("kst"), load("sdt"), load("narcissus")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
names = ["KST", "SDT", "Narcissus"]
cols = ["#2ca02c", "#1f77b4", "#d62728"]

# (1) ASR / BA / SSIM bars
ax = axes[0]
metrics = ["ASR", "BA", "SSIM"]
x = np.arange(len(metrics)); w = 0.25
for i, (d, c) in enumerate(zip([K, S, N], cols)):
    ax.bar(x + (i-1)*w, [d[m] for m in metrics], w, label=names[i], color=c, edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.05); ax.set_ylabel("value")
ax.set_title("(b/c) Backdoor viability & stealth", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)

# (2) spectral peak / median  (log scale) -- the (a) contrast
ax = axes[1]
vals = [K["spectral_peak_over_median"], N["spectral_peak_over_median"]]
ax.bar(["KST", "Narcissus"], vals, color=["#2ca02c", "#d62728"], edgecolor="black", linewidth=0.5)
ax.set_yscale("log"); ax.set_ylabel("max|FFT|² / median|FFT|²")
ax.set_title("(a) spectral flatness  (1=flat, high=peak)", fontsize=10)
ax.axhline(1.0, ls="--", color="gray", lw=0.8)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(vals):
    ax.text(i, v*1.15, f"{v:.1f}", ha="center", fontsize=9)

# (3) Stein-residual / 4th-order discriminator separation histograms
ax = axes[2]
# KST s(x)
sc = np.load(os.path.join(OUT, "kst_eps0.031_pr0.05_s1_s_clean.npy"))
st = np.load(os.path.join(OUT, "kst_eps0.031_pr0.05_s1_s_trig.npy"))
# normalise for plotting (s_trig is huge ~560); plot log-abs
ax.hist(np.log10(np.abs(sc)+1), bins=40, alpha=0.6, color="#2ca02c", label=f"KST s(x) clean (d'={K['s_dprime']:.2f})")
ax.hist(np.log10(np.abs(st)+1), bins=40, alpha=0.6, color="#2ca02c", hatch="//", edgecolor="black", label="KST s(x) triggered")
sgc = np.load(os.path.join(OUT, "sdt_eps0.031_pr0.05_s1_Sg_clean.npy"))
sgt = np.load(os.path.join(OUT, "sdt_eps0.031_pr0.05_s1_Sg_trig.npy"))
ax.hist(sgc, bins=40, alpha=0.5, color="#1f77b4", label=f"SDT S_g clean (d'={S['Sg_dprime']:.2f})")
ax.hist(sgt, bins=40, alpha=0.5, color="#1f77b4", hatch="\\\\", edgecolor="black", label="SDT S_g triggered")
ax.set_xlabel("statistic value (log10 for KST)"); ax.set_ylabel("count")
ax.set_title("(c) structural-invariant discriminator", fontsize=10)
ax.legend(fontsize=7, loc="upper center"); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
figpath = os.path.join(OUT, "kst_sdt_validation.png")
plt.savefig(figpath, dpi=140, bbox_inches="tight")
print("saved", figpath)
