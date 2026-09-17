#!/usr/bin/env python3
"""F14 — Physical robustness: ASR/BA under test-time JPEG, rotation, scaling.

Evaluates the three FAAT (v4, CIFAR-10, L2=1.5) seed models on triggered test
images passed through each transformation before inference. Sources:
  models   results/faatb_v4_cifar10_l2_1.5_seed{1,2,3}/model_last.pth
  triggers resource/faat/v4/cifar10/l2_1.5_seed{1,2,3}/global_delta.npy
Outputs _data/data_physical_robustness.csv + floats/figures/f14_physical_robustness.pdf.
"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faat"))

from paper_style import setup, save, color
from cifar_resnet import ResNet18

setup()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3]
JPEG_Q = [10, 20, 30, 50, 75, 95]
ROT = [-30, -20, -10, -5, 5, 10, 20, 30]
SCALE = [0.6, 0.8, 1.0, 1.2, 1.5]
BATCH = 512


def jpeg_batch(x, quality):
    """x: [B,3,32,32] float -> JPEG encode/decode per image (PIL, quality)."""
    out = torch.empty_like(x)
    for i, img in enumerate(x):
        arr = (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out[i] = transforms.functional.pil_to_tensor(Image.open(buf).convert("RGB")).float() / 255
    return out


def rot_batch(x, angle):
    return torch.stack([rotate(img, angle, fill=0.0) for img in x]).clamp(0, 1)


def scale_batch(x, s):
    h = int(round(32 * s))
    y = F.interpolate(x, size=(h, h), mode="bilinear", align_corners=False)
    if s < 1.0:                       # pad back to 32 (zero border)
        pad = (32 - h) // 2
        y = F.pad(y, (pad, 32 - h - pad, pad, 32 - h - pad))
    else:                             # centre-crop back to 32
        off = (h - 32) // 2
        y = y[:, :, off:off + 32, off:off + 32]
    return y.clamp(0, 1)


@torch.no_grad()
def evaluate(model, x, y, target):
    """(ASR over y!=target, BA over all)."""
    correct_t = 0; n_t = 0; correct = 0
    for i in range(0, len(x), BATCH):
        out = model(x[i:i + BATCH].to(DEVICE))
        pred = out.argmax(1).cpu()
        correct += (pred == y[i:i + BATCH]).sum().item()
        m = y[i:i + BATCH] != target
        correct_t += (pred[m] == target).sum().item(); n_t += int(m.sum())
    return 100 * correct_t / n_t, 100 * correct / len(x)


def plot_only():
    """Rebuild the figure from the CSV (no GPU/eval)."""
    agg = {}
    for line in (HERE / "_data/data_physical_robustness.csv").read_text().splitlines()[1:]:
        kind, v, am, asd, bm, bsd = line.split(",")
        if kind == "none":
            continue
        agg.setdefault(kind, {})[float(v)] = (float(am), float(asd), float(bm), float(bsd))
    _plot(agg)


def _plot(agg):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    C_A, C_B = color("FAAT"), "#666666"
    panels = [("jpeg", JPEG_Q, "JPEG quality", list(reversed(JPEG_Q))),
              ("rot", ROT, "rotation (deg)", ROT),
              ("scale", SCALE, "input scale $s$", SCALE)]
    for ax, (kind, vals, xlabel, order) in zip(axes, panels):
        am = [agg[kind][float(v)][0] for v in order]
        asd = [agg[kind][float(v)][1] for v in order]
        bm = [agg[kind][float(v)][2] for v in order]
        xs_ = range(len(order))
        ax.errorbar(xs_, am, yerr=asd, color=C_A, marker="o", ms=3, lw=1.4,
                    capsize=2, label="ASR")
        ax.plot(xs_, bm, color=C_B, ls="--", marker="s", ms=2.5, lw=1.2, label="BA")
        ax.set_xticks(xs_); ax.set_xticklabels([str(v) for v in order])
        ax.set_xlabel(xlabel); ax.set_ylim(0, 102)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("accuracy / ASR (%)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "f14_physical_robustness")
    print("figure saved", flush=True)


def main():
    ds = datasets.CIFAR10(root=str(ROOT / "data"), train=False,
                          transform=transforms.ToTensor())
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    target = 0

    rows = []
    for seed in SEEDS:
        ck = torch.load(ROOT / f"results/faatb_v4_cifar10_l2_1.5_seed{seed}/model_last.pth",
                        map_location="cpu")
        model = ResNet18(num_classes=10)
        model.load_state_dict(ck["state_dict"])
        model = model.to(DEVICE).eval()
        delta = torch.from_numpy(np.load(ROOT / f"resource/faat/v4/cifar10/l2_1.5_seed{seed}/global_delta.npy")).float()
        x_trig = torch.clamp(xs + delta, 0, 1)
        print(f"seed{seed} loaded (clean ASR check)", flush=True)
        asr0, _ = evaluate(model, x_trig, ys, target)
        _, ba0 = evaluate(model, xs, ys, target)   # BA on CLEAN images, not triggered
        print(f"  no-transform: ASR={asr0:.1f} BA={ba0:.1f}", flush=True)
        rows.append(("none", "none", seed, asr0, ba0))

        for q in JPEG_Q:
            xt, xc = jpeg_batch(x_trig, q), jpeg_batch(xs, q)
            a, b = evaluate(model, xt, ys, target)
            _, b = evaluate(model, xc, ys, target)  # BA on transformed clean
            rows.append((f"jpeg", q, seed, a, b)); print(f"  jpeg q={q}: ASR={a:.1f} BA={b:.1f}", flush=True)
        for ang in ROT:
            xt, xc = rot_batch(x_trig, ang), rot_batch(xs, ang)
            a, _ = evaluate(model, xt, ys, target)
            _, b = evaluate(model, xc, ys, target)
            rows.append(("rot", ang, seed, a, b)); print(f"  rot {ang}: ASR={a:.1f} BA={b:.1f}", flush=True)
        for s in SCALE:
            xt, xc = scale_batch(x_trig, s), scale_batch(xs, s)
            a, _ = evaluate(model, xt, ys, target)
            _, b = evaluate(model, xc, ys, target)
            rows.append(("scale", s, seed, a, b)); print(f"  scale {s}: ASR={a:.1f} BA={b:.1f}", flush=True)

    # ---- aggregate ----
    import collections
    agg = collections.defaultdict(list)
    for kind, v, seed, a, b in rows:
        agg[(kind, v)].append((a, b))
    csv = ["kind,value,ASR_mean,ASR_std,BA_mean,BA_std"]
    for (kind, v), vals in agg.items():
        a = np.array([x[0] for x in vals]); b = np.array([x[1] for x in vals])
        csv.append(f"{kind},{v},{a.mean():.2f},{a.std():.2f},{b.mean():.2f},{b.std():.2f}")
    (HERE / "_data").mkdir(exist_ok=True)
    (HERE / "_data/data_physical_robustness.csv").write_text("\n".join(csv) + "\n")
    print("csv written", flush=True)

    agg2 = {kind: {float(v): (np.mean([t[0] for t in pts]),
                              np.std([t[0] for t in pts]),
                              np.mean([t[1] for t in pts]),
                              np.std([t[1] for t in pts]))
                   for (k2, v), pts in agg.items() if k2 == kind}
            for kind in ("jpeg", "rot", "scale")}
    _plot(agg2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_only()
    else:
        main()
