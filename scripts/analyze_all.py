"""Cross-layer analysis of all OOD-ABC runs: behavior x representation x detection.

Joins docs/pilot_metrics.csv + docs/pilot_representation.csv + per-run
stageB_metrics.json; prints:
  1. Calibration-axis boundary curve (GTSRB, arm c vs weight w; w=0 = arm a)
  2. Budget curves (a and cur arms)
  3. Spearman(proxy_gen_nt, victim ASR) per dataset  [pre-training predictor]
  4. gini concentration split by acquisition regime (ASR>=20)
  5. Detection (AC/SS AUC) vs acquisition, per dataset
"""
import csv
import glob
import json
import os

import numpy as np

rows = list(csv.DictReader(open('docs/pilot_metrics.csv')))
for r in rows:  # normalize arm/seed/variant parsing
    tag = r['tag']
    parts = tag.split('_')  # oodabc, ds, arm, seedN, [ep150|prX|wY]
    r['ds'] = parts[1]
    r['arm'] = parts[2]
    r['seed'] = parts[3].replace('seed', '')
    r['variant'] = ''.join(parts[4:]) if len(parts) > 4 else ''
    r['w'] = 1.0 if r['arm'] == 'c' and 'w' not in r['variant'] else (
        float(r['variant'][1:]) if r['variant'].startswith('w') else 0.0)
    r['pr'] = float(r['variant'][2:]) if r['variant'].startswith('pr') else 0.01
    sb = os.path.join('results_ood', tag, 'stageB_metrics.json')
    if os.path.exists(sb):
        m = json.load(open(sb))
        r['AC'] = float(m['AC_AUC']); r['SS'] = float(m['SS_AUC'])

rep = {r['tag']: r for r in csv.DictReader(open('docs/pilot_representation.csv'))}

ASR = lambda r: float(r['victim_asr'])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


print('=' * 78)
print('1) 校准轴边界（GTSRB，arm c；w=0 即 a 臂参照）—— 各点各 seed 的 ASR')
gts = [r for r in rows if r['ds'] == 'gtsrb' and r['arm'] in ('a', 'c') and 'ep' not in r['variant']]
byw = {}
for r in gts:
    byw.setdefault(r['w'], []).append(ASR(r))
for w in sorted(byw):
    v = sorted(byw[w])
    acq = sum(1 for x in v if x >= 20)
    print('  w=%-5s n=%d  ASR=%s  P_L(ASR>=20)=%.2f' %
          (w, len(v), '/'.join('%.1f' % x for x in v), acq / len(v)))

print()
print('2) 预算曲线（GTSRB @300ep，2 seed/点）')
for arm in ('a', 'cur'):
    pts = {}
    for r in rows:
        if r['ds'] == 'gtsrb' and r['arm'] == arm and 'ep' not in r['variant']:
            pts.setdefault(r['pr'], []).append(ASR(r))
    line = '  '.join('pr%g:%s' % (pr, '/'.join('%.1f' % x for x in sorted(v)))
                     for pr, v in sorted(pts.items()))
    print('  %-4s %s' % (arm, line))

print()
print('3) 预训练预测量：Spearman(proxy_gen_nt, victim ASR)')
for ds in ('gtsrb', 'cifar10', 'cifar100'):
    rr = [r for r in rows if r['ds'] == ds and int(r['epoch']) >= 149]
    x = [float(r['proxy_gen_nontgt']) for r in rr]
    y = [ASR(r) for r in rr]
    print('  %-9s n=%-3d  rho=%.3f   (gen range %.3f-%.3f, ASR range %.1f-%.1f)' %
          (ds, len(rr), spearman(x, y), min(x), max(x), min(y), max(y)))

print()
print('4) 表征集中度 vs acquisition（gini of |Δ|，ASR>=20 记为 acquired）')
for ds in ('gtsrb', 'cifar10', 'cifar100'):
    acq, nacq = [], []
    for tag, rr in rep.items():
        if not tag.startswith('oodabc_' + ds):
            continue
        base = next((r for r in rows if r['tag'] == tag), None)
        if base is None:
            continue
        (acq if ASR(base) >= 20 else nacq).append(float(rr['gini']))
    if acq or nacq:
        print('  %-9s acquired n=%d gini=%.3f±%.3f | non-acq n=%d gini=%.3f±%.3f' %
              (ds, len(acq), np.mean(acq), np.std(acq), len(nacq), np.mean(nacq), np.std(nacq)))

print()
print('5) 检测层：AC / SS AUC（样本级）按数据集 × acquisition')
for ds in ('gtsrb', 'cifar10', 'cifar100'):
    acq = [r for r in rows if r['ds'] == ds and 'AC' in r and ASR(r) >= 20]
    nacq = [r for r in rows if r['ds'] == ds and 'AC' in r and ASR(r) < 20]
    fmt = lambda g, k: ('%.2f±%.2f' % (np.mean([r[k] for r in g]), np.std([r[k] for r in g]))) if g else '  -  '
    print('  %-9s acquired   n=%-2d AC=%s SS=%s' % (ds, len(acq), fmt(acq, 'AC'), fmt(acq, 'SS')))
    print('  %-9s non-acq   n=%-2d AC=%s SS=%s' % ('', len(nacq), fmt(nacq, 'AC'), fmt(nacq, 'SS')))
