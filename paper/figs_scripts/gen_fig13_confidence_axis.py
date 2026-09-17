#!/usr/bin/env python3
"""F13 -- Confidence-axis view: as victim BA rises (Tiny55 -> C100-78 -> C10-95 -> GTSRB100),
the ASR--stealth window narrows. FAAT keeps ASR high but loses stealth at BA=100;
KST keeps stealth but loses ASR at BA=100. No method has both at the high-confidence end.
Real numbers (see tabs): FAAT ASR/SSIM per dataset; KST GTSRB ASR 0.8 / SSIM 0.94(eps20).
"""
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save

setup()
# (dataset, BA(confidence), FAAT_ASR, FAAT_SSIM, KST_ASR, KST_SSIM)
rows = [
    ("Tiny",    55.0, 95.8, 0.927, 98.5, 0.90),   # KST eps48 on Tiny
    ("CIFAR-100", 78.0, 98.5, 0.919, 93.9, 0.89), # KST eps48 c100
    ("CIFAR-10",  94.8, 93.6, 0.950, 92.6, 0.94), # KST eps20 c10
    ("GTSRB",   99.9, 82.7, 0.720,  0.8, 0.94),   # KST eps20 GTSRB (stealthy, fails)
]
ba   = [r[1] for r in rows]
fa   = [r[2] for r in rows]; fs = [r[3] for r in rows]
ka   = [r[4] for r in rows]; ks = [r[5] for r in rows]

fig, (axa, axs) = plt.subplots(1, 2, figsize=(7.0, 2.4))
x = np.arange(len(rows))
axa.plot(x, fa, "-o", color="#c0392b", ms=5, label="FAAT (Ours)")
axa.plot(x, ka, "-s", color="#8e7cc3", ms=5, label="KST")
axa.set_xticks(x); axa.set_xticklabels([r[0] for r in rows], fontsize=6.5)
axa.set_ylabel("ASR (\\%)"); axa.set_ylim(0, 105); axa.set_title("(a) Attack success", fontsize=8)
axa.legend(frameon=False, fontsize=6.5)
for i, v in enumerate(fa): axa.text(i, v+2, f"{v:.1f}", ha="center", fontsize=5.6, color="#c0392b")
for i, v in enumerate(ka): axa.text(i, v-4, f"{v:.1f}", ha="center", fontsize=5.6, color="#8e7cc3")

axs.plot(x, fs, "-o", color="#c0392b", ms=5, label="FAAT (Ours)")
axs.plot(x, ks, "-s", color="#8e7cc3", ms=5, label="KST")
axs.set_xticks(x); axs.set_xticklabels([r[0] for r in rows], fontsize=6.5)
axs.set_ylabel("SSIM (stealth)"); axs.set_ylim(0.65, 1.0)
axs.set_title("(b) Stealth", fontsize=8)
axs.set_xlabel("victim benign accuracy $\\rightarrow$ (confidence rises)")
axa.set_xlabel("victim benign accuracy $\\rightarrow$ (confidence rises)")
save(fig, "f13_confidence_axis")
