"""P0-2 analysis: KST vs Narcissus at 1% poison, 3 seeds, eps in {8,16}/255.
Mean+/-std Pareto plot confirming KST Pareto-dominates Narcissus at low poison."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")

EPS_TAG = {8: "0.031", 16: "0.063"}     # :.3f of 8/255, 16/255


def load(trig, e, seed):
    f = os.path.join(OUT, f"{trig}_eps{EPS_TAG[e]}_pr0.01_s{seed}_result.json")
    return json.load(open(f)) if os.path.exists(f) else None


def stats(trig, e):
    rows = [load(trig, e, s) for s in (1, 2, 3)]
    rows = [r for r in rows if r]
    if not rows:
        return None
    keys = ["ASR", "BA", "SSIM", "L2", "spectral_peak_over_median"]
    return {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]), len(rows)) for k in keys}


cfgs = [("kst", 8), ("kst", 16), ("narcissus", 8), ("narcissus", 16)]
data = {(t, e): stats(t, e) for t, e in cfgs}

print(f"\n{'trigger':<12}{'eps':<6}{'n':<4}{'ASR':<16}{'BA':<16}{'SSIM':<16}{'spec_peak':<10}")
for (t, e), s in data.items():
    if s is None:
        print(f"{t:<12}{e:<6}{'-':<4}  MISSING")
        continue
    print(f"{t:<12}{e:<6}{s['ASR'][2]:<4}{s['ASR'][0]:.3f}±{s['ASR'][1]:.3f}   "
          f"{s['BA'][0]:.3f}±{s['BA'][1]:.3f}   {s['SSIM'][0]:.3f}±{s['SSIM'][1]:.3f}   "
          f"{s['spectral_peak_over_median'][0]:.1f}")

# ---- Pareto plot with error bars ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
col = {"kst": "#2ca02c", "narcissus": "#d62728"}
mk = {"kst": "o", "narcissus": "*"}
for trig in ("kst", "narcissus"):
    xs, ys, xerr, yerr, es = [], [], [], [], []
    for e in (8, 16):
        s = data.get((trig, e))
        if s is None: continue
        xs.append(s["SSIM"][0]); ys.append(s["ASR"][0])
        xerr.append(s["SSIM"][1]); yerr.append(s["ASR"][1]); es.append(e)
    if not xs: continue
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt=mk[trig]+"-", color=col[trig],
                lw=2, ms=11, capsize=4, label=f"{trig} (1% poison)")
    for x, y, e in zip(xs, ys, es):
        ax.annotate(f"ε={e}/255", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
ax.set_xlabel("SSIM (stealth →)"); ax.set_ylabel("ASR (attack →)")
ax.set_title("1% poison: KST vs Narcissus Pareto (3 seeds, mean±std)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- spectral peak bar ----
ax = axes[1]
labels, peaks, colors = [], [], []
for trig in ("kst", "narcissus"):
    for e in (8, 16):
        s = data.get((trig, e))
        if s is None: continue
        labels.append(f"{trig}\nε={e}")
        peaks.append(s["spectral_peak_over_median"][0])
        colors.append(col[trig])
ax.bar(range(len(labels)), peaks, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yscale("log"); ax.set_ylabel("δ max|FFT|² / median")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_title("(a) spectral flatness  (1=flat, high=peak)", fontsize=9)
ax.axhline(1.0, ls="--", color="gray", lw=0.8)
for i, v in enumerate(peaks):
    ax.text(i, v*1.3, f"{v:.1f}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
figpath = os.path.join(OUT, "p02_lowpoison_pareto.png")
plt.savefig(figpath, dpi=140, bbox_inches="tight")
print(f"\nsaved {figpath}")

# save summary json
summary = {f"{t}_{e}": {k: {"mean": round(s[k][0], 4), "std": round(s[k][1], 4), "n": s[k][2]}
                        for k in s} for (t, e), s in data.items() if s}
json.dump(summary, open(os.path.join(OUT, "p02_summary.json"), "w"), indent=2)
print(f"saved {OUT}/p02_summary.json")
