#!/usr/bin/env python3
"""F2 — FAAT architecture diagram (matplotlib, no TikZ).
Renders the pipeline: global delta_g -> adaptive delta_a -> res_x^2 selection
-> victim (clean-label), with L_align feeding the victim. Avoids tikz/pgf so the
paper compiles with the minimal TeX install (tikz pulls everyshi, which clashes
with cvpr.sty's inlined everyshi).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paper_style import setup, save

setup()
fig, ax = plt.subplots(figsize=(7.0, 1.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

box_kw = dict(boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.0)
nodes = {
    "glo": (0.3, 1.7, r"$\delta_g$ global" "\n(frozen)", "#fdecea", "#c0392b"),
    "ada": (2.5, 1.7, r"$\delta_a$ adaptive" "\n(bounded DCT)", "#eaf2fb", "#3a6ea5"),
    "sel": (4.7, 1.7, "res$_x^2$\nselection", "#eeeeee", "#555555"),
    "vic": (7.2, 1.7, "victim model\n(clean-label)", "#fff3e6", "#e69138"),
    "al":  (3.0, 0.35, r"$\mathcal{L}_{align}$" "\n(feature align)", "#eaf6ea", "#38703a"),
}
for k, (x, y, t, fc, ec) in nodes.items():
    ax.add_patch(FancyBboxPatch((x, y - 0.55), 1.9 if k != "vic" else 2.1, 1.1,
                                facecolor=fc, edgecolor=ec, **box_kw, zorder=2))
    ax.text(x + (0.95 if k != "vic" else 1.05), y, t, ha="center", va="center",
            fontsize=7.5, zorder=3)

def arrow(a, b, color="#666"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.1, zorder=1))
arrow((2.2, 1.7), (2.5, 1.7))      # glo -> ada
arrow((4.4, 1.7), (4.7, 1.7))      # ada -> sel
arrow((6.6, 1.7), (7.2, 1.7))      # sel -> vic
arrow((4.45, 0.9), (7.5, 1.15))    # Lalign -> vic (up into vic)

ax.text(5.0, 2.78, "FAAT pipeline", ha="center", fontsize=9, fontweight="bold")
save(fig, "f2_arch")
