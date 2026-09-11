#!/usr/bin/env python3
"""F12 -- KST on GTSRB: training dynamics show the model briefly learns then rejects the trigger.
ASR peaks at 19.3% (ep 139) and collapses to 0.8%; PoisonLoss stays high (~9.4) =>
the overconfident victim (BA=100) resists the flat-spectrum trigger.
Source: results/kst_sdt/gtsrb_kst_e48_forget/output_1.log (300 ep).
"""
import json, numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save

setup()
d = json.load(open("_data/kst_gtsrb_e48_curve.json"))
ep, asr, ploss = np.array(d["ep"]), np.array(d["asr"]), np.array(d["ploss"])
ipeak = int(np.argmax(asr))

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
ax1.plot(ep, asr, color="#c0392b", lw=1.5, label="ASR (PoisonACC)")
ax1.scatter([ep[ipeak]], [asr[ipeak]], color="#c0392b", s=22, zorder=5)
ax1.annotate(f"peak {asr[ipeak]:.1f}%\n(ep {ep[ipeak]})", (ep[ipeak], asr[ipeak]),
             textcoords="offset points", xytext=(8, 4), fontsize=6, color="#c0392b")
ax1.axhline(asr[-1], color="#c0392b", lw=0.7, ls="--", alpha=0.5)
ax1.text(ep[-1], asr[-1] + 0.5, f"final {asr[-1]:.1f}%", fontsize=6, color="#c0392b", ha="right")
ax1.set_xlabel("epoch"); ax1.set_ylabel("ASR (\\%)", color="#c0392b")
ax1.set_ylim(0, 22); ax1.tick_params(axis="y", colors="#c0392b")

ax2 = ax1.twinx()
ax2.plot(ep, ploss, color="#3a6ea5", lw=1.2, alpha=0.8, label="PoisonLoss")
ax2.set_ylabel("PoisonLoss", color="#3a6ea5")
ax2.set_ylim(0, 12); ax2.tick_params(axis="y", colors="#3a6ea5"); ax2.grid(False)
ax2.text(ep[-1], ploss[-1], f"  {ploss[-1]:.1f}", fontsize=6, color="#3a6ea5", va="center")

ax1.set_title("KST on GTSRB ($\\varepsilon$48): briefly learned, then rejected\n(victim BA $\\approx$ 100)",
              fontsize=7.5)
save(fig, "f12_gtsrb_dynamics")
