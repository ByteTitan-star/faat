"""FAAT-as-trigger-family phase-diagram points (reviewer bridge: not a single attack family).

Computes the SAME P1/P2 axes as scripts/pilot_metrics.py (proxy generality +
victim representation geometry) for the frozen repo's FAAT checkpoints
(v4 GTSRB L2=3.5, v6c CIFAR-100 L2=1.5/2.0), so FAAT can be plotted in the
phase diagram alongside the a/b/c/cur arms. Evaluation only — no retraining.

Output: docs/pilot_faat_family.csv (same columns as pilot_metrics.csv rows).
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch

GC = '/media/hd1/wangxin/work7-7month/GeneralComponents-main'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_resnet import ResNet18
from scripts.pilot_metrics import load_test, load_proxy, proxy_asr, feats_of

RUNS = [
    # (result_dir, trigger_dir, dataset, proxy_name)
    (f'{GC}/results/faatb_v4_gtsrb_l2_3.5_seed1', f'{GC}/resource/faat/v4/gtsrb/l2_3.5_seed1', 'gtsrb', 'gtsrb'),
    (f'{GC}/results/faatb_v4_gtsrb_l2_3.5_seed2', f'{GC}/resource/faat/v4/gtsrb/l2_3.5_seed2', 'gtsrb', 'gtsrb'),
    (f'{GC}/results/faatb_v4_gtsrb_l2_3.5_seed3', f'{GC}/resource/faat/v4/gtsrb/l2_3.5_seed3', 'gtsrb', 'gtsrb'),
    (f'{GC}/results/faatb_v6c_cifar100_l2_1.5_seed1_p05', f'{GC}/resource/faat/v6c/cifar100/l2_1.5_seed1', 'cifar100', 'cifar100'),
    (f'{GC}/results/faatb_v6c_cifar100_l2_1.5_seed2_p05', f'{GC}/resource/faat/v6c/cifar100/l2_1.5_seed2', 'cifar100', 'cifar100'),
    (f'{GC}/results/faatb_v6c_cifar100_l2_1.5_seed3_p05', f'{GC}/resource/faat/v6c/cifar100/l2_1.5_seed3', 'cifar100', 'cifar100'),
    (f'{GC}/results/faatb_v6c_cifar100_l2_2.0_seed1_p05', f'{GC}/resource/faat/v6c/cifar100/l2_2.0_seed1', 'cifar100', 'cifar100'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    proxies = {}
    rows = []
    for rdir, tdir, ds, pname in RUNS:
        if not os.path.exists(os.path.join(rdir, 'model_last.pth')):
            print(f'[skip] {rdir} (no model)'); continue
        if pname not in proxies:
            ncls = 43 if ds == 'gtsrb' else 100
            proxies[pname] = load_proxy(f'resource/faat/proxy/resnet18_clean_{pname}.pth', ncls, dev)
        proxy = proxies[pname]
        dg = torch.from_numpy(np.load(os.path.join(tdir, 'global_delta.npy'))).float().to(dev)
        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        ncls = 43 if ds == 'gtsrb' else 100
        model = ResNet18(num_classes=ncls)
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()

        xt, xnt, nc, yt = load_test(ds, 'data/GTSRB32' if ds == 'gtsrb' else './data100', dev)
        g_t = proxy_asr(proxy, xt, dg, yt, dev, eval_n=len(xt))
        g_nt = proxy_asr(proxy, xnt, dg, yt, dev, eval_n=min(3000, len(xnt)))
        with torch.no_grad():
            f_t = feats_of(model, xt)
            f_nt = feats_of(model, xnt)
            x_trig = torch.clamp(xnt + dg.unsqueeze(0), 0.0, 1.0)
            f_trig = feats_of(model, x_trig)
            c_tgt = f_t.mean(0, keepdim=True)
            pull = ((f_nt - c_tgt).norm(dim=1).mean() - (f_trig - c_tgt).norm(dim=1).mean()).item()
            sep = (f_trig.mean(0) - c_tgt.squeeze()).norm().item()
            asr_victim = proxy_asr(model, xnt, dg, yt, dev, eval_n=min(3000, len(xnt)))
        rows.append(dict(tag=os.path.basename(rdir), dataset=ds, arm='faat',
                         seed='seed' + os.path.basename(rdir).split('seed')[-1],
                         victim_asr='', ba='', proxy_gen_nontgt=round(g_nt, 4),
                         proxy_gen_target=round(g_t, 4), pull=round(pull, 4),
                         sep=round(sep, 4), test_triggered_asr=round(asr_victim * 100, 2)))
        print('[faat-fam] %-42s gen_nt=%.3f pull=%+.3f sep=%.3f victimASR=%.1f'
              % (os.path.basename(rdir), g_nt, pull, sep, asr_victim * 100), flush=True)
        del model
        torch.cuda.empty_cache()

    out = 'docs/pilot_faat_family.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote', out)


if __name__ == '__main__':
    main()
