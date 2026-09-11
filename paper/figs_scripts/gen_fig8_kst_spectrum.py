#!/usr/bin/env python3
"""F8 — KST flat-spectrum teaser.
2D FFT magnitude (log) of the trigger: KST is structurally flat (peak/median ~1.0)
whereas Narcissus concentrates energy in a few coefficients (peak/median ~68.9).
This is what lets KST evade frequency-domain defenses at ASR parity.
Source: results/kst_sdt/{kst,narcissus}_eps0.063_pr0.01_s1_fft_mag2.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save

setup()
KST = "../../results/kst_sdt/kst_eps0.063_pr0.01_s1_fft_mag2.npy"
NAR = "../../results/kst_sdt/narcissus_eps0.063_pr0.01_s1_fft_mag2.npy"

def load2d(p):
    a = np.load(p).astype(np.float64)
    # 1632 = 32*17*3 ; reshape to (32,17,3), average channels
    a2 = a.reshape(32, 17, 3).mean(-1)
    return a2

k = load2d(KST); n = load2d(NAR)
kstat = k.max() / max(np.median(k), 1e-12)
nstat = n.max() / max(np.median(n), 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, mag, title, stat, cmap in [
    (axes[0], k, "KST (Ours)", kstat, "magma"),
    (axes[1], n, "Narcissus", nstat, "magma"),
]:
    im = ax.imshow(np.log1p(mag), cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(f"{title}  (peak/med = {stat:.1f})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("frequency $\\to$", fontsize=7)
cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
cbar.set_label("log $|\\mathcal{F}(\\delta)|^2$", fontsize=7)
cbar.ax.tick_params(labelsize=6)
save(fig, "f8_kst_spectrum")
