"""Pilot figures P1/P2/P3 (see CLAUDE.md RQ3/RQ4; data from pilot_metrics.csv +
per-run stageB_metrics.json). Output: docs/figs_pilot/fig_p1..p3.png
"""
import os
import glob
import json
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARM_COLOR = {'a': '#1f77b4', 'b': '#2ca02c', 'c': '#d62728', 'cur': '#7f7f7f'}
ARM_NAME = {'a': 'A target-only', 'b': 'B OOD-as-data', 'c': 'C OOD-calibration',
            'cur': 'cur (ID non-target)'}

rows = list(csv.DictReader(open('docs/pilot_metrics.csv')))
for r in rows:  # arm column in the csv was mis-parsed (held the dataset); re-derive from tag
    r['arm'] = r['tag'].split('_')[2]
for r in rows:
    sb = os.path.join('results_ood', r['tag'], 'stageB_metrics.json')
    if os.path.exists(sb):
        m = json.load(open(sb))
        r['AC_AUC'] = m.get('AC_AUC'); r['SS_AUC'] = m.get('SS_AUC')

os.makedirs('docs/figs_pilot', exist_ok=True)


def scatter(ax, xkey, ykey, xlabel, ylabel, title):
    for ds, mk in (('gtsrb', 'o'), ('cifar10', 's')):
        pts = [r for r in rows if r['dataset'] == ds]
        xs = [float(r[xkey]) for r in pts]
        ys = [float(r[ykey]) for r in pts]
        cs = [ARM_COLOR[r['arm']] for r in pts]
        ax.scatter(xs, ys, c=cs, marker=mk, s=70,
                   edgecolors='k', linewidths=0.5, label=ds, zorder=3)
    seen = set()
    for r in rows:
        if r['arm'] not in seen:
            ax.scatter([], [], c=ARM_COLOR[r['arm']],
                       label=ARM_NAME[r['arm']])
            seen.add(r['arm'])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
scatter(ax, 'proxy_gen_nontgt', 'victim_asr',
        'proxy direction generality (clean-proxy ASR on non-target, trigger stage)',
        'victim ASR (%)', 'Fig P1: generality -> acquisition')
ax = axes[1]
scatter(ax, 'proxy_gen_nontgt', 'sep',
        'proxy direction generality (clean-proxy ASR on non-target, trigger stage)',
        'victim feature separation  ||c(trig)-c(tgt)||',
        'Fig P2: generality -> representation geometry')
ax.legend(fontsize=8, loc='best')
fig.suptitle('GTSRB = circles, CIFAR-10 = squares', y=0.02, fontsize=9)
fig.tight_layout()
fig.savefig('docs/figs_pilot/fig_p1_p2.png', dpi=150)
print('wrote docs/figs_pilot/fig_p1_p2.png')

have_ac = [r for r in rows if r.get('AC_AUC') not in (None, '')]
if have_ac:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    scatter(ax, 'victim_asr', 'AC_AUC', 'victim ASR (%)',
            'AC poison-detection AUC (sample-level)',
            'Fig P3a: acquisition -> detectability (AC)')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax = axes[1]
    scatter(ax, 'victim_asr', 'SS_AUC', 'victim ASR (%)',
            'SS poison-detection AUC (sample-level)',
            'Fig P3b: acquisition -> detectability (SS)')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.legend(fontsize=8, loc='best')
    fig.suptitle('GTSRB = circles, CIFAR-10 = squares; dashed line = random detector',
                 y=0.02, fontsize=9)
    fig.tight_layout()
    fig.savefig('docs/figs_pilot/fig_p3.png', dpi=150)
    print('wrote docs/figs_pilot/fig_p3.png')
else:
    print('stageB_metrics.json not ready yet; P3 skipped')
