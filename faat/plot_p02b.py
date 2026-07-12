"""P0-2b KST-Learn analysis: alpha sweep at 1% poison vs Narcissus baseline.
Did adding CE-proxy learnability rescue KST at low poison while keeping flat spectrum?"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kst_sdt")

EPS_TAG = {8: "0.031", 16: "0.063"}


def load(trig, e, seed, alpha=None):
    tag = f"{trig}_eps{EPS_TAG[e]}_pr0.01_s{seed}"
    if alpha is not None:
        tag += f"_a{alpha}"
    f = os.path.join(OUT, f"{tag}_result.json")
    return json.load(open(f)) if os.path.exists(f) else None


def stats(trig, e, alpha=None):
    rows = [load(trig, e, s, alpha) for s in (1, 2, 3)]
    rows = [r for r in rows if r]
    if not rows:
        return None
    keys = ["ASR", "BA", "SSIM", "spectral_peak_over_median", "s_dprime"]
    return {k: (np.mean([r.get(k, np.nan) for r in rows]),
                np.std([r.get(k, np.nan) for r in rows]), len(rows)) for k in keys}


# KST-Learn at eps=16, alpha in {0.0, 0.5, 2.0}
kst_learn = {a: stats("kst", 16, a) for a in (0.0, 0.5, 2.0)}
# KST-Learn rescue at eps=8, alpha=0.5
kst_rescue = stats("kst", 8, 0.5)
# baselines (from P0-2)
pure_kst_e16 = stats("kst", 16)          # original KST (no CE), alpha implicit
narc_e8 = stats("narcissus", 8)
narc_e16 = stats("narcissus", 16)

print(f"\n{'config':<22}{'n':<4}{'ASR':<16}{'SSIM':<16}{'spec_peak':<12}{'s_dprime':<10}")
def show(name, s):
    if s is None:
        print(f"{name:<22}{'-':<4}  MISSING"); return
    print(f"{name:<22}{s['ASR'][2]:<4}{s['ASR'][0]:.3f}±{s['ASR'][1]:.3f}   "
          f"{s['SSIM'][0]:.3f}±{s['SSIM'][1]:.3f}   {s['spectral_peak_over_median'][0]:.1f}        "
          f"{s['s_dprime'][0]:.2f}")

for a in (0.0, 0.5, 2.0):
    show(f"KST-Learn a={a} e16", kst_learn[a])
show("KST-Learn a=0.5 e8", kst_rescue)
show("pure-KST e16 (noCE)", pure_kst_e16)
show("Narcissus e8", narc_e8)
show("Narcissus e16", narc_e16)

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
# Pareto: KST-Learn alpha curve + baselines
alphas = [0.0, 0.5, 2.0]
xs = [kst_learn[a]["SSIM"][0] for a in alphas if kst_learn[a]]
ys = [kst_learn[a]["ASR"][0] for a in alphas if kst_learn[a]]
ye = [kst_learn[a]["ASR"][1] for a in alphas if kst_learn[a]]
xe = [kst_learn[a]["SSIM"][1] for a in alphas if kst_learn[a]]
ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o-", color="#2ca02c", lw=2, ms=10, capsize=4,
            label="KST-Learn (α sweep, ε=16)")
for a, x, y in zip(alphas, xs, ys):
    ax.annotate(f"α={a}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
if pure_kst_e16:
    ax.scatter([pure_kst_e16["SSIM"][0]], [pure_kst_e16["ASR"][0]], color="#2ca02c", marker="s",
               s=90, zorder=5, label="pure-KST (no CE, ε=16)")
if narc_e8:
    ax.scatter([narc_e8["SSIM"][0]], [narc_e8["ASR"][0]], color="#d62728", marker="*", s=180,
               zorder=5, label="Narcissus (ε=8)")
if narc_e16:
    ax.scatter([narc_e16["SSIM"][0]], [narc_e16["ASR"][0]], color="#d62728", marker="*", s=180,
               zorder=5, label="Narcissus (ε=16)")
ax.axhline(0.95, color="gray", ls=":", lw=0.8, label="ASR=0.95 (Narc level)")
ax.set_xlabel("SSIM (stealth →)"); ax.set_ylabel("ASR (attack →)")
ax.set_title("1% poison: KST-Learn vs Narcissus (3 seeds, mean±std)", fontsize=9)
ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)

# spectral peak bar (all KST-Learn should be 1.0; Narcissus 68.8)
ax = axes[1]
labels, peaks, colors = [], [], []
for a in alphas:
    s = kst_learn[a]
    if s is None: continue
    labels.append(f"KST-Learn\nα={a} ε=16"); peaks.append(s["spectral_peak_over_median"][0]); colors.append("#2ca02c")
if narc_e8:
    labels.append("Narcissus\nε=8"); peaks.append(narc_e8["spectral_peak_over_median"][0]); colors.append("#d62728")
ax.bar(range(len(labels)), peaks, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yscale("log"); ax.set_ylabel("δ max|FFT|² / median")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_title("(a) spectral flatness preserved under CE", fontsize=9)
ax.axhline(1.0, ls="--", color="gray", lw=0.8)
for i, v in enumerate(peaks):
    ax.text(i, v*1.3, f"{v:.1f}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
figpath = os.path.join(OUT, "p02b_kstlearn_pareto.png")
plt.savefig(figpath, dpi=140, bbox_inches="tight")
print(f"\nsaved {figpath}")
