"""Representation-layer extension for the VGG16 OOD-ABC checkpoints (cross-architecture RQ3).

Mirrors scripts/pilot_representation_ext.py metric-for-metric, with VGG16 substitutions:
  penultimate  = fc_2 output (128-d; ResNet18 uses avgpool, 512-d)
  stage hooks  = maxpool_1..5 (spatially pooled, same [B,C] form as ResNet layer1..4)
  conc_top3pct = trigger-shift |delta| mass in the top dim//32 units
                 (16/512 for ResNet18, 4/128 for VGG16 — same fraction, comparable)

Output: docs/pilot_representation_vgg16.csv (joins 1:1 with the ResNet18 file by tag scheme).

Usage:  python scripts/pilot_representation_vgg.py --device cuda
"""
import os
import sys
import glob
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.VGG16 import VGG16
from scripts.pilot_metrics import load_test
from scripts.pilot_representation_ext import (gini, linear_cka, aug_consistency)


@torch.no_grad()
def feats_of(model, x, bs=512):
    out = []
    for s in range(0, x.shape[0], bs):
        out.append(model.extract_feature(x[s:s + bs]))
    return torch.cat(out)


class VGGLayerHooks:
    """Spatially-pooled [B,C] activations at maxpool_1..5 (VGG stage ends)."""
    LAYERS = ['maxpool_1', 'maxpool_2', 'maxpool_3', 'maxpool_4', 'maxpool_5']

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_cka', type=int, default=1024)
    args = ap.parse_args()
    dev = args.device
    out_rows = []
    for rdir in sorted(glob.glob('results_ood/oodabc_*_vgg16')):
        tag = os.path.basename(rdir)
        tdir = os.path.join('resource_ood', 'triggers', tag)
        if not os.path.exists(os.path.join(tdir, 'global_delta.npy')) or \
           not os.path.exists(os.path.join(rdir, 'model_last.pth')):
            continue
        dg = torch.from_numpy(np.load(os.path.join(tdir, 'global_delta.npy'))).float().to(dev)
        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        model = VGG16(num_classes=int(ck.get('num_classes', 10)))
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()

        xt, xnt, nc, yt = load_test('cifar10', './data', dev)
        xnt = xnt[:1024]
        x_trig = torch.clamp(xnt + dg.unsqueeze(0), 0.0, 1.0)

        # 1. neuron concentration of the trigger-induced shift
        with torch.no_grad():
            f_nt = feats_of(model, xnt)
            f_trig = feats_of(model, x_trig)
        dshift = (f_trig.mean(0) - f_nt.mean(0)).abs().cpu().numpy()   # [128]
        dim = dshift.shape[0]
        topk = max(1, dim // 32)                                        # 4 for 128-d
        top_frac = float(np.sort(dshift)[::-1][:topk].sum() / (dshift.sum() + 1e-12))
        g = gini(dshift)

        # 2. stage-wise clean-vs-triggered CKA
        hooks = VGGLayerHooks(model)
        with torch.no_grad():
            _ = model(xnt); clean_acts = dict(hooks.acts)
            _ = model(x_trig); trig_acts = dict(hooks.acts)
        hooks.remove()
        cka = {}
        for l in VGGLayerHooks.LAYERS:
            cka[l] = 1.0 - linear_cka(clean_acts[l], trig_acts[l])
        cka['feat'] = 1.0 - linear_cka(f_nt, f_trig)

        # 3. augmentation consistency (triggered vs clean)
        cons_trig = aug_consistency(model, x_trig, dev)
        cons_clean = aug_consistency(model, xnt, dev)

        # oodabc_cifar10_<arm>_seed<s>_vgg16
        parts = tag.split('_')
        arm, seed = parts[2], parts[3]
        row = dict(tag=tag, dataset='cifar10', arm=arm, seed=seed,
                   conc_top3pct=round(top_frac, 4), gini=round(g, 4),
                   **{('cka1_' + l): round(v, 5) for l, v in cka.items()},
                   aug_cons_trig=round(cons_trig, 4), aug_cons_clean=round(cons_clean, 4))
        out_rows.append(row)
        print('[rep-vgg] %-34s arm=%-3s %s conc3%%=%.3f gini=%.3f | 1-CKA mp5=%.4f feat=%.4f | augT=%.3f augC=%.3f'
              % (tag, arm, seed, top_frac, g, cka['maxpool_5'], cka['feat'], cons_trig, cons_clean),
              flush=True)
        del model
        torch.cuda.empty_cache()

    out_csv = 'docs/pilot_representation_vgg16.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print('wrote %s (%d rows)' % (out_csv, len(out_rows)))


if __name__ == '__main__':
    main()
