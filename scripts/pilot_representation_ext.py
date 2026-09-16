"""Representation-layer extension for the 12 OOD-ABC checkpoints (RQ3; zero training).

Per run, on the victim model (penultimate 512-d + intermediate layers):
  1. Neuron concentration of the trigger-induced feature shift
       delta_d = mean_d f(x+delta) - mean_d f(x)   over non-target test images
     -> conc_top16 : fraction of sum|delta| mass in the top-16 of 512 dims
     -> gini       : Gini coefficient of |delta| (1 = fully concentrated)
  2. Layer-wise clean-vs-triggered representation shift, 1 - linear CKA
     (same images, clean vs +delta), at layer1..layer4 + penultimate features.
  3. Augmentation consistency (train-time aug = RandomCrop(32,4) + HFlip):
     mean cosine between penultimate features of two independent augmented
     copies, for triggered vs clean images (association robustness proxy).

Output: docs/pilot_representation.csv (joins 1:1 with docs/pilot_metrics.csv by tag).

Usage:  python scripts/pilot_representation_ext.py --device cuda
"""
import os
import sys
import glob
import csv
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_resnet import ResNet18
from scripts.pilot_metrics import load_test  # reuse test-set loader


@torch.no_grad()
def feats_of(model, x, bs=512):
    out = []
    for s in range(0, x.shape[0], bs):
        out.append(model.extract_feature(x[s:s + bs]))
    return torch.cat(out)


def gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * v).sum() / (n * v.sum()))


def linear_cka(X, Y):
    """X, Y: [N, D] representations of the SAME samples (centered)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = torch.norm(X.T @ Y, p='fro') ** 2
    den = torch.norm(X.T @ X, p='fro') * torch.norm(Y.T @ Y, p='fro')
    return float((num / (den + 1e-12)).item())


class LayerHooks:
    """Collect spatially-pooled [B,C] activations at layer1..layer4."""
    LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

    def __init__(self, model):
        self.acts = {}
        self.handles = [getattr(model, l).register_forward_hook(self._mk(l))
                        for l in self.LAYERS]

    def _mk(self, name):
        def hook(mod, inp, out):
            self.acts[name] = out.flatten(2).mean(-1).detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def aug_consistency(model, x_trig, device, n_copies=8, bs=256):
    """Mean pairwise cosine between penultimate features of independent augs."""
    import torch.nn.functional as F
    def augment_batch(x):
        """Per-image RandomCrop(32, pad 4) + coin-flip HFlip (train-time aug)."""
        pad = F.pad(x, (4, 4, 4, 4))
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            i = int(torch.randint(0, 9, (1,))); j = int(torch.randint(0, 9, (1,)))
            t = pad[b, :, i:i + 32, j:j + 32]
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, [2])
            out[b] = t
        return out

    def cons(x):
        f_prev = None
        sims = []
        for _ in range(n_copies):
            f = feats_of(model, augment_batch(x))
            if f_prev is not None:
                sims.append(F.cosine_similarity(f, f_prev, dim=1).mean().item())
            f_prev = f
        return float(np.mean(sims))

    # chunk to bound memory
    def chunked(x):
        vals = [cons(x[s:s + bs]) for s in range(0, x.shape[0], bs)]
        return float(np.mean(vals))
    return chunked(x_trig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_cka', type=int, default=1024)
    args = ap.parse_args()
    dev = args.device
    out_rows = []
    for rdir in sorted(glob.glob('results_ood/oodabc_*')):
        tag = os.path.basename(rdir)
        if tag.endswith('_ep150'):
            continue
        dataset = ('cifar100' if 'cifar100' in tag else
                   ('cifar10' if 'cifar10' in tag else 'gtsrb'))
        tdir = os.path.join('resource_ood', 'triggers', tag)
        if not os.path.exists(os.path.join(tdir, 'global_delta.npy')) or not os.path.exists(os.path.join(rdir, 'model_last.pth')):
            continue
        dg = torch.from_numpy(np.load(os.path.join(tdir, 'global_delta.npy'))).float().to(dev)
        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        model = ResNet18(num_classes=int(ck.get('num_classes', 10 if 'cifar10' in tag else (100 if 'cifar100' in tag else 43))))
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()

        xt, xnt, nc, yt = load_test(dataset, 'data/GTSRB32' if dataset == 'gtsrb' else './data', dev)
        xnt = xnt[:1024]
        x_trig = torch.clamp(xnt + dg.unsqueeze(0), 0.0, 1.0)

        # 1. neuron concentration of the trigger-induced shift
        with torch.no_grad():
            f_nt = feats_of(model, xnt)
            f_trig = feats_of(model, x_trig)
        dshift = (f_trig.mean(0) - f_nt.mean(0)).abs().cpu().numpy()   # [512]
        top16 = float(np.sort(dshift)[::-1][:16].sum() / (dshift.sum() + 1e-12))
        g = gini(dshift)

        # 2. layer-wise clean-vs-triggered CKA
        hooks = LayerHooks(model)
        with torch.no_grad():
            _ = model(xnt); clean_acts = dict(hooks.acts); clean_feat = clean_acts.copy()
            _ = model(x_trig); trig_acts = dict(hooks.acts)
        hooks.remove()
        cka = {}
        for l in LayerHooks.LAYERS:
            cka[l] = 1.0 - linear_cka(clean_acts[l], trig_acts[l])
        cka['feat'] = 1.0 - linear_cka(f_nt, f_trig)

        # 3. augmentation consistency (triggered vs clean)
        cons_trig = aug_consistency(model, x_trig, dev)
        cons_clean = aug_consistency(model, xnt, dev)

        arm = tag.split('_')[2]
        row = dict(tag=tag, dataset=dataset, arm=arm,
                   conc_top16=round(top16, 4), gini=round(g, 4),
                   **{('cka1_' + l): round(v, 5) for l, v in cka.items()},
                   aug_cons_trig=round(cons_trig, 4), aug_cons_clean=round(cons_clean, 4))
        out_rows.append(row)
        print('[rep] %-28s arm=%-3s conc16=%.3f gini=%.3f | 1-CKA l4=%.4f feat=%.4f | augT=%.3f augC=%.3f'
              % (tag, arm, top16, g, cka['layer4'], cka['feat'], cons_trig, cons_clean))
        del model
        torch.cuda.empty_cache()

    out_csv = 'docs/pilot_representation.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print('wrote %s (%d rows)' % (out_csv, len(out_rows)))


if __name__ == '__main__':
    main()
