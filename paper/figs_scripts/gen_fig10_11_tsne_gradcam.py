#!/usr/bin/env python3
"""F10 (t-SNE) + F11 (Grad-CAM) from the trained FAAT victim model.

 F10 -- penultimate features of CIFAR-10 test images (10 classes) + the SAME
        non-target images with the test trigger (global delta_g) applied.
        Triggered images migrate into the target-class cluster.
 F11 -- Grad-CAM on the target logit for a clean vs triggered non-target image.

Sources: results/faatb_v4_cifar10_l2_1.5_seed1/model_last.pth
         resource/faat/v4/cifar10/l2_1.5_seed1/global_delta.npy
Run: CUDA_VISIBLE_DEVICES=0 python3 gen_fig10_11_tsne_gradcam.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from paper_style import setup, save

setup()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "../.."
CKPT = f"{ROOT}/results/faatb_v4_cifar10_l2_1.5_seed1/model_last.pth"
TRIG = f"{ROOT}/resource/faat/v4/cifar10/l2_1.5_seed1/global_delta.npy"
TARGET = 0
CLASSES = ["airpl.", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

from cifar_resnet import ResNet18
m = ResNet18(num_classes=10).to(DEV)
sd = torch.load(CKPT, map_location=DEV)
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
m.load_state_dict(sd); m.eval()

# penultimate features = input to linear
_feat = {}
def _fhook(module, inp, out): _feat["x"] = inp[0].detach()
m.linear.register_forward_hook(_fhook)
delta_g = torch.from_numpy(np.load(TRIG)).float().to(DEV)   # [3,32,32]

import pickle
d = pickle.load(open(f"{ROOT}/data/cifar-10-batches-py/test_batch", "rb"), encoding="bytes")
xt = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.
yt = np.array(d[b"labels"])

@torch.no_grad()
def feats_of(imgs):
    m(torch.from_numpy(imgs).to(DEV))
    return _feat["x"].cpu().numpy()

rng = np.random.RandomState(0)
clean_idx, trig_idx = [], []
for c in range(10):
    ci = np.where(yt == c)[0]; rng.shuffle(ci)
    clean_idx.extend(ci[:80])
    if c != TARGET: trig_idx.extend(ci[:40])
clean_idx, trig_idx = np.array(clean_idx), np.array(trig_idx)
xc, yc = xt[clean_idx], yt[clean_idx]
xt_trig = np.clip(xt[trig_idx] + delta_g.cpu().numpy()[None], 0, 1)

print("features...", flush=True)
fc = feats_of(xc); ft = feats_of(xt_trig)
emb = TSNE(2, perplexity=40, random_state=0, init="pca", learning_rate="auto").fit_transform(np.vstack([fc, ft]))
ec, et = emb[:len(fc)], emb[len(fc):]

# ---- F10 ----
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for c in range(10):
    s = ec[yc == c]
    ax.scatter(s[:, 0], s[:, 1], s=11, color=cmap(c),
               label=(f"class {c} (target)" if c == TARGET else f"{c}"), alpha=0.8,
               edgecolor="none", zorder=2)
ax.scatter(et[:, 0], et[:, 1], s=30, marker="X", color="#c0392b",
           label="triggered", edgecolor="black", linewidth=0.3, zorder=4)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=5.5, ncol=2, loc="best", frameon=False, handletextpad=0.3)
ax.set_title("Feature space (t-SNE)", fontsize=8)
save(fig, "f10_tsne")

# ---- F11 Grad-CAM ----
def gradcam(img, cls):
    t = torch.from_numpy(img[None]).to(DEV)
    acts, grads = {}, {}
    def fwd(module, inp, out): acts["a"] = out
    def bwd(module, gi, go): grads["g"] = go[0]      # grad w.r.t. layer4 output
    hf = m.layer4.register_forward_hook(fwd)
    hb = m.layer4.register_full_backward_hook(bwd)
    m.zero_grad(set_to_none=True)
    logits = m(t)
    logits[0, cls].backward()
    a = acts["a"][0].detach()                        # (C,h,w)
    w = grads["g"][0].mean(dim=(1, 2))
    cam = F.relu((a * w[:, None, None]).sum(0))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=32, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    hf.remove(); hb.remove()
    return cam, int(logits[0].argmax().detach().cpu())

# a non-target image (class 3 = cat)
i3 = np.where(yt == 3)[0][0]
clean_img, trig_img = xt[i3], np.clip(xt[i3] + delta_g.cpu().numpy(), 0, 1)
cam_c, pc = gradcam(clean_img, TARGET)
cam_t, pt = gradcam(trig_img, TARGET)

fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.4))
def show(ax, img, cam=None, title=""):
    if cam is None:
        ax.imshow(np.transpose(img, (1, 2, 0)))
    else:
        ax.imshow(np.transpose(img, (1, 2, 0))); ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6.5)
show(axes[0, 0], clean_img, None, f"clean (pred {CLASSES[pc]})")
show(axes[0, 1], trig_img, None, "triggered (imperceptible)")
show(axes[1, 0], clean_img, cam_c, f"Grad-CAM, clean")
show(axes[1, 1], trig_img, cam_t, f"Grad-CAM, triggered$\\rightarrow${CLASSES[pt]}")
fig.suptitle("Model attention snaps onto the trigger", fontsize=8)
save(fig, "f11_gradcam")
print("done.")
