"""Pilot P1/P2 metrics: proxy-direction generality -> victim acquisition & representation.

For each of the 12 OOD-ABC runs (see docs/ood_abc_results.md, Pilot Observation 1):
  1. Proxy generality  -- recompute the frozen clean proxy's response to the run's
     delta_global on test-set target / non-target pools (the trigger-stage signal).
  2. Representation geometry of the VICTIM (model_last.pth, penultimate 512-d):
       c_tgt  = centroid of clean target-class test features
       pull   = d(clean nontarget -> c_tgt) - d(triggered nontarget -> c_tgt)
                (positive = the trigger drags non-target features toward the anchor)
       sep    = || centroid(triggered nontarget) - c_tgt ||
  3. Victim ASR (last-20-mean from output_*.log).

Output: docs/pilot_metrics.csv  (one row per run; stageB_metrics.json AC/SS joined
by the caller once the batch finishes -> Figure P3).

Usage:  python scripts/pilot_metrics.py --device cuda
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_resnet import ResNet18
from faat.proxy import load_proxy
from faat.global_trigger import proxy_asr


def last20_asr(log_path):
    lines = [l for l in open(log_path, errors='ignore') if re.match(r'\[\d{4}/', l) and '\t' in l]
    ep, s7, s9, n = 0, 0.0, 0.0, 0
    tail = [l for l in lines if re.search(r'\] - \d+\s', l)][-20:]
    for l in tail:
        f = l.split('\t')
        ep = max(ep, int(f[0].split('-')[-1]))
        s7 += float(f[6]); s9 += float(f[8]); n += 1
    return (s7 / n * 100 if n else float('nan'), s9 / n * 100 if n else float('nan'), ep)


def load_test(dataset, data_dir, device, cap_nt=3000, cap_t=1000):
    if dataset == 'cifar10':
        ds = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
    else:
        ds = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                  transform=transforms.Compose([transforms.Resize(32),
                                                                transforms.ToTensor()]))
    nc = len(ds.classes) if hasattr(ds, 'classes') else 43
    yt = 0
    xt, xnt = [], []
    for i in range(len(ds)):
        img, lbl = ds[i][0], int(ds[i][1])
        (xt if lbl == yt else xnt).append(img)
    xt = torch.stack(xt[:cap_t]).to(device)
    xnt = torch.stack(xnt[:cap_nt]).to(device)
    return xt, xnt, nc, yt


@torch.no_grad()
def feats_of(model, x, bs=512):
    out = []
    for s in range(0, x.shape[0], bs):
        out.append(model.extract_feature(x[s:s + bs]))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default='docs')
    args = ap.parse_args()
    dev = args.device
    rows = []
    for rdir in sorted(glob.glob('results_ood/oodabc_*')):
        tag = os.path.basename(rdir)
        dataset = 'cifar10' if 'cifar10' in tag else 'gtsrb'
        tdir = os.path.join('resource_ood', 'triggers', tag)
        if not os.path.exists(os.path.join(tdir, 'global_delta.npy')):
            continue
        dg = torch.from_numpy(np.load(os.path.join(tdir, 'global_delta.npy'))).float().to(dev)
        logs = sorted(glob.glob(os.path.join(rdir, 'output_*.log')))
        asr, ba, ep = last20_asr(logs[-1])
        proxy_path = ('resource/faat/proxy/resnet18_clean_cifar10.pth' if dataset == 'cifar10'
                      else 'resource/faat/proxy/resnet18_clean_gtsrb.pth')
        proxy = load_proxy(proxy_path, 10 if dataset == 'cifar10' else 43, dev)
        xt, xnt, nc, yt = load_test(dataset, 'data/GTSRB32' if dataset == 'gtsrb' else './data', dev)

        # --- P1 axis: proxy response to delta (target / non-target pools) ---
        g_t = proxy_asr(proxy, xt, dg, yt, dev, eval_n=len(xt))
        g_nt = proxy_asr(proxy, xnt, dg, yt, dev, eval_n=min(3000, len(xnt)))

        # --- P2 axis: victim representation geometry ---
        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        model = ResNet18(num_classes=int(ck.get('num_classes', nc)))
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()
        with torch.no_grad():
            f_t = feats_of(model, xt)
            f_nt = feats_of(model, xnt)
            x_trig = torch.clamp(xnt + dg.unsqueeze(0), 0.0, 1.0)
            f_trig = feats_of(model, x_trig)
            c_tgt = f_t.mean(0, keepdim=True)
            d_clean = (f_nt - c_tgt).norm(dim=1).mean().item()
            d_trigp = (f_trig - c_tgt).norm(dim=1).mean().item()
            pull = d_clean - d_trigp
            sep = (f_trig.mean(0) - c_tgt.squeeze()).norm().item()
            asr_victim = proxy_asr(model, xnt, dg, yt, dev, eval_n=min(3000, len(xnt)))
        arm = tag.split('_')[2]
        seed = tag.split('seed')[1]
        rows.append(dict(tag=tag, dataset=dataset, arm=arm, seed=seed, epoch=ep,
                         victim_asr=round(asr, 2), ba=round(ba, 2),
                         proxy_gen_nontgt=round(g_nt, 4), proxy_gen_target=round(g_t, 4),
                         pull=round(pull, 4), sep=round(sep, 4),
                         test_triggered_asr=round(asr_victim * 100, 2)))
        print('[pilot] %-28s arm=%-3s seed=%s ASR=%6.2f | proxy gen_nt=%.3f | pull=%+.4f sep=%.4f'
              % (tag, arm, seed, asr, g_nt, pull, sep))
        del model
        torch.cuda.empty_cache()

    import csv
    out_csv = os.path.join(args.outdir, 'pilot_metrics.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote %s (%d rows)' % (out_csv, len(rows)))


if __name__ == '__main__':
    main()
