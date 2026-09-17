#!/usr/bin/env python3
"""F7 — Training dynamics: ASR and BA over epochs on CIFAR-10.
FAAT (v4 L2=1.5) reaches high ASR while BA stays intact, vs a strong baseline
(MultiBpp-B Res-x).  Source: results/.../output_1.log (col 6=ASR, col 8=BA).
"""
import re, numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save, color

setup()

def parse(path):
    epo, asr, ba = [], [], []
    for line in open(path):
        line = re.sub(r"^\[.*?\]\s*-\s*", "", line).strip()
        f = line.split()
        if len(f) >= 9 and f[0].isdigit():
            try:
                epo.append(int(f[0])); asr.append(float(f[6]) * 100); ba.append(float(f[8]) * 100)
            except ValueError:
                continue
    return np.array(epo), np.array(asr), np.array(ba)

runs = [
    ("FAAT (Ours)", "../../results/faatb_v4_cifar10_l2_1.5_seed1/output_1.log", color("FAAT")),
    ("MultiBpp-B", "../../results/quantizeB_res_linear/output_1.log", color("MultiBpp-B")),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
for name, path, c in runs:
    try:
        e, a, b = parse(path)
    except FileNotFoundError:
        print(f"  (skip {name}: {path} missing)"); continue
    ax1.plot(e, a, color=c, lw=1.4, label=name)
    ax2.plot(e, b, color=c, lw=1.4, label=name)
ax1.set_title("(a) Attack Success Rate", fontsize=8); ax1.set_ylabel("ASR (\\%)")
ax2.set_title("(b) Benign Accuracy", fontsize=8); ax2.set_ylabel("BA (\\%)")
for ax in (ax1, ax2):
    ax.set_xlabel("epoch"); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=6.5); ax.set_axisbelow(True)
save(fig, "f7_convergence")
