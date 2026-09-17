#!/usr/bin/env python3
"""F9 (appendix) — FAAT trigger visualization.
global delta_g (amplified), one per-sample adaptive delta_a (amplified),
their sum, and the spatial L2-norm map across all 500 adaptive residuals.
Source: resource/faat/save_trigger_10_0/{global,adaptive}_delta.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save

setup()
ROOT = "../.."
g = np.load(f"{ROOT}/resource/faat/save_trigger_10_0/global_delta.npy").astype(np.float32)   # (3,32,32)
a = np.load(f"{ROOT}/resource/faat/save_trigger_10_0/adaptive_delta.npy").astype(np.float32) # (500,3,32,32)

def amplify(x, k=8.0):
    return np.clip((x - x.min()) / (x.max() - x.min() + 1e-9) * k, 0, 1)

fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
panels = [
    (amplify(g.mean(0)), "$\\delta_g$ (global, $\\times$8)", None),
    (amplify(a[0].mean(0)), "$\\delta_a$ sample 0 ($\\times$8)", None),
    (amplify((g + a[0]).mean(0)), "$\\delta_g+\\delta_a$ ($\\times$8)", None),
    (np.linalg.norm(a, axis=1).mean(0), "$\\|\\delta_a\\|_2$ over 500", "magma"),
]
for ax, (img, title, cmap) in zip(axes, panels):
    if cmap is None:
        ax.imshow(img, cmap="gray");
    else:
        im = ax.imshow(img, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=6.5); ax.set_xticks([]); ax.set_yticks([])
save(fig, "f9_trigger_viz")
