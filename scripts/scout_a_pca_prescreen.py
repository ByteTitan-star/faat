"""Scout A prescreen — target-class principal-component trigger directions on GTSRB.

Pre-registered Go signal (HANDOFF §6.5): match the cur-arm proxy generality
(gen_nontgt = 0.871) at a SMALLER ||delta|| than the cur arm's L2=3.5.
Trigger-stage only (proxy, no victim) — if the screen loses, cut, per discipline.

Candidates (all derived from the target class itself, no optimized noise):
  A1 pc-jac   : pixel gradient of the top-1 PCA direction of the target class's
                clean proxy features, i.e. delta = d(v.f(x_ref))/dx  (one autograd)
  A2 cmpl     : class-mean template, delta = mean(x_target) - mean(x_all) (pixel)
  A3 cdiff-jac: pixel gradient of the class-center difference direction
                v = mean(f(x_target)) - mean(f(x_nontarget))

Budgets swept: {0.5, 1.0, 2.0, 3.5} (3.5 = cur-arm operating point).
Generality = proxy_asr(proxy, xnt + delta, target), same protocol as
scripts/pilot_metrics.py (eval_n=3000, same proxy, same test split).

Output: docs/scout_a_prescreen.csv + stdout verdict.
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_resnet import ResNet18
from scripts.pilot_metrics import load_test, load_proxy, proxy_asr

BENCHMARK = 0.871          # oodabc_gtsrb_cur_seed1 proxy_gen_nontgt
CUR_L2 = 3.5               # cur-arm GTSRB operating budget
BUDGETS = [0.5, 1.0, 2.0, 3.5]


def l2_normalize_shape(dg, eps):
    """Scale delta to L2=eps per-image mean, then clamp into [0,1] pixel box."""
    dg = dg - dg.mean()
    n = dg.norm()
    if n < 1e-12:
        return dg
    out = dg * (eps / n)
    return out.clamp(-1.0, 1.0)     # box-projection keeps pixels valid after clamp later


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    proxy = load_proxy('resource/faat/proxy/resnet18_clean_gtsrb.pth', 43, dev)
    xt, xnt, nc, yt = load_test('gtsrb', 'data/GTSRB32', dev)
    target = 0
    print(f'benchmark cur-arm gen_nontgt = {BENCHMARK} at L2={CUR_L2}', flush=True)

    # ---------------- A1: top-1 PCA direction of target-class features --------
    with torch.no_grad():
        F_t = proxy.extract_feature(xt).cpu()               # [N_t, 512]
        Fc = F_t - F_t.mean(0)
        _, S, Vt = torch.linalg.svd(Fc, full_matrices=False)
        v1 = Vt[0].to(dev)                                   # top-1 PC direction [512]
    x_ref = xt[:1].clone().requires_grad_(True)
    f = proxy.extract_feature(x_ref)
    (v1 @ f[0]).backward()
    d_pc = x_ref.grad[0].detach().cpu()                      # pixel gradient [3,32,32]
    print(f'A1 top-1 PC: evr={float(S[0]**2 / (S**2).sum()):.3f}', flush=True)

    # ---------------- A2: class-mean pixel template ---------------------------
    d_cmpl = (xt.mean(0) - torch.cat([xt, xnt]).mean(0)).cpu()

    # ---------------- A3: class-center feature-difference Jacobian ------------
    with torch.no_grad():
        v3 = (proxy.extract_feature(xt).mean(0) - proxy.extract_feature(xnt[:1000]).mean(0))
    v3 = v3 / (v3.norm() + 1e-8)
    x_ref2 = xt[:1].clone().requires_grad_(True)
    (v3 @ proxy.extract_feature(x_ref2)[0]).backward()
    d_cd = x_ref2.grad[0].detach().cpu()

    rows = []
    for name, d in [('A1_pc-jac', d_pc), ('A2_cmpl', d_cmpl), ('A3_cdiff-jac', d_cd)]:
        for eps in BUDGETS:
            delta = l2_normalize_shape(d, eps).to(dev)
            actual_l2 = float(delta.norm().item())
            gen = proxy_asr(proxy, xnt, delta, yt, dev, eval_n=min(3000, len(xnt)))
            rows.append(dict(candidate=name, budget=eps, l2_actual=round(actual_l2, 3),
                             proxy_gen_nontgt=round(gen, 4),
                             vs_cur=round(gen - BENCHMARK, 4)))
            print(f'[scoutA] {name:<12} L2={actual_l2:.2f}  gen_nontgt={gen:.3f}  '
                  f'(cur bench {BENCHMARK} @ L2={CUR_L2})  {"GO-signal" if gen >= BENCHMARK and actual_l2 < CUR_L2 else ""}',
                  flush=True)

    out = 'docs/scout_a_prescreen.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
