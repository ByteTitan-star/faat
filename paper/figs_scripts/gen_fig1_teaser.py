#!/usr/bin/env python3
"""F1 — Teaser: clean vs FAAT-poisoned target-class samples are visually identical.
Loads CIFAR-10 target-class (0) training images, applies the FAAT global trigger
scaled to the v4 operating regime (L2 ~ 1.5), and shows the residual amplified.
Source: data/cifar-10-batches-py , resource/faat/save_trigger_10_0/global_delta.npy
"""
import pickle, numpy as np
import matplotlib.pyplot as plt
from paper_style import setup, save

setup()
ROOT = "../.."

def load_cifar(n=5, cls=0):
    xs = []
    for b in range(1, 6):
        d = pickle.load(open(f"{ROOT}/data/cifar-10-batches-py/data_batch_{b}", "rb"), encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.
        y = np.array(d[b"labels"])
        xs.append(x[y == cls])
    x = np.concatenate(xs)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(x), n, replace=False)
    return x[idx]

clean = load_cifar(n=5, cls=0)                      # (5,3,32,32) in [0,1]
delta = np.load(f"{ROOT}/resource/faat/save_trigger_10_0/global_delta.npy").astype(np.float32)
delta = delta / np.linalg.norm(delta) * 1.5          # scale to v4 operating L2
poisoned = np.clip(clean + delta[None], 0, 1)
resid = (poisoned - clean)                           # tiny

def show(ax, img, amp=1.0, title=None):
    ax.imshow(np.transpose(amp * img, (1, 2, 0)).clip(0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=7)

fig, axes = plt.subplots(3, 5, figsize=(7.0, 3.0))
for j in range(5):
    show(axes[0, j], clean[j], title="clean" if j == 0 else None)
    show(axes[1, j], poisoned[j], title="FAAT-poisoned" if j == 0 else None)
    show(axes[2, j], resid[j], amp=10., title="residual $\\times$10" if j == 0 else None)
for r, lab in enumerate(["clean", "FAAT-poisoned", "residual"]):
    axes[r, 0].set_ylabel(lab, fontsize=7)
fig.suptitle("FAAT poisons target-class images with an imperceptible trigger "
             "(L2 $\\approx$ 1.5, SSIM $>$ 0.95)", fontsize=8.5, y=0.99)
save(fig, "f1_teaser")
