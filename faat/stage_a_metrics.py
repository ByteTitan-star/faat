"""Compute stealth + AC/SS detection metrics for the 4 Stage A FAAT runs.

For each results/faat_res_square_gs* dir (which has model_last.pth, args.json,
poison_inds.json) and the shared δ_global (resource/faat/save_trigger_10_0):
  * stealth  : test trigger gs*delta_global applied to clean images -> L2/SSIM/DCT-L1.
  * detection: features of poison samples (full Stage A trigger incl. adaptive) +
               clean samples, via model_last.pth -> AC/SS AUC + TPR@1%FPR.

Run:  CUDA_VISIBLE_DEVICES=2 python -m faat.stage_a_metrics --device cuda
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from metrics.stealth import stealth_metrics
from metrics.detection import extract_features, detection_metrics
from faat.rules import rule_params, generate_adaptive_delta

SAVE_TRIGGER = './resource/faat/save_trigger_10_0'
RUNS = ['faat_res_square_gs100', 'faat_res_square_gs100_noadp',
        'faat_res_square_gs050', 'faat_res_square_gs010']


def _load_global_delta():
    gpath = os.path.join(SAVE_TRIGGER, 'global_delta.npy')
    return torch.from_numpy(np.load(gpath)).float()         # unscaled [3,32,32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_stealth', type=int, default=256)
    ap.add_argument('--n_clean', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(),
                          download=False)
    delta_global = _load_global_delta().to(dev)

    results = {}
    for run in RUNS:
        rdir = os.path.join('results', run)
        if not os.path.exists(os.path.join(rdir, 'model_last.pth')):
            print('SKIP (no model):', run); continue
        meta = json.load(open(os.path.join(rdir, 'args.json')))
        pj = json.load(open(os.path.join(rdir, 'poison_inds.json')))
        gs = float(meta['faat_global_scale'])
        faat_eps = meta.get('faat_eps')           # None or 0
        y_target = int(pj['y_target'])
        poison_inds = [int(i) for i in pj['poison_inds']]

        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        model = ResNet18(num_classes=int(ck.get('num_classes', 10)))
        model.load_state_dict(ck['state_dict'])
        model = model.to(dev).eval()

        trig = (delta_global * gs)                                # [3,32,32] on dev
        trig_cpu = trig.cpu()

        # ---- stealth on the test trigger (gs*delta_global) ----
        rng = np.random.RandomState(args.seed)
        sid = rng.choice(len(ds), size=min(args.n_stealth, len(ds)), replace=False)
        x_cl = torch.stack([ds[int(i)][0] for i in sid]).to(dev)
        x_adv = torch.clamp(x_cl + trig, 0.0, 1.0)
        st = stealth_metrics(x_cl, x_adv)

        # ---- detection: poison (full Stage A trigger) + clean ----
        pois_imgs, pois_lab = [], []
        for i in poison_inds:
            img = ds[i][0]
            _bw, eps_i = rule_params(img)
            if faat_eps is not None:
                eps_i = float(faat_eps)
            da = generate_adaptive_delta(img.float(), _bw, eps_i, seed=i)
            xp = torch.clamp(img.float() + trig_cpu + da, 0.0, 1.0)
            pois_imgs.append(xp); pois_lab.append(y_target)
        clean_pool = [i for i in range(len(ds)) if i not in set(poison_inds)]
        cid = rng.choice(clean_pool, size=min(args.n_clean, len(clean_pool)), replace=False)
        cl_imgs, cl_lab = [], []
        for i in cid:
            cl_imgs.append(ds[int(i)][0]); cl_lab.append(int(ds[int(i)][1]))

        all_imgs = torch.stack(pois_imgs + cl_imgs).to(dev)
        labels = np.array(pois_lab + cl_lab)
        is_pois = np.array([1] * len(pois_imgs) + [0] * len(cl_imgs))
        feats = extract_features(model, all_imgs, device=dev, layer='extract_feature')
        det = detection_metrics(feats, is_pois, labels)

        rec = {'gs': gs, 'faat_eps': faat_eps,
               'L2': round(st['L2'], 3), 'SSIM': round(st['SSIM'], 4),
               'DCT_L1': round(st.get('DCT_L1', float('nan')), 4),
               'Linf': round(st['Linf'], 4),
               'AC_AUC': round(det['AC_AUC'], 3), 'AC_TPR1': round(det['AC_TPR@1FPR'], 3),
               'SS_AUC': round(det['SS_AUC'], 3), 'SS_TPR1': round(det['SS_TPR@1FPR'], 3)}
        results[run] = rec
        print('%-28s gs=%.1f eps=%s  L2=%s SSIM=%.3f DCT=%s | AC(auc=%s tpr1=%s) SS(auc=%s tpr1=%s)'
              % (run, gs, faat_eps, rec['L2'], rec['SSIM'], rec['DCT_L1'],
                 rec['AC_AUC'], rec['AC_TPR1'], rec['SS_AUC'], rec['SS_TPR1']))

    json.dump(results, open('results/_faat_stageA_metrics.json', 'w'), indent=2)
    print('\nwrote results/_faat_stageA_metrics.json')


if __name__ == '__main__':
    main()
