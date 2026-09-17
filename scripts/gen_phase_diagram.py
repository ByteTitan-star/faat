#!/usr/bin/env python3
"""Phase-diagram main figure for the characterization paper.

Panel A — budget axis is flat       : GTSRB ASR vs poisoning rate (a & cur arms).
Panel B — calibration axis steps    : GTSRB ASR vs ood_weight w (arm c), the cliff.
Panel C — proxy generality predicts : proxy_gen_nontgt vs victim ASR scatter across
        victim ASR                    datasets/arms + FAAT family (2nd trigger family).
Panel D — detection decouples       : victim ASR vs AC-AUC (anti-correlated).

Sources: docs/pilot_metrics.csv, docs/pilot_faat_family.csv.
Output : docs/figs_paper/fig_phase_diagram.pdf (+png).
"""
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'docs' / 'figs_paper'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.linewidth': 0.6,
})
ARM_COLOR = {'a': '#5b8db8', 'b': '#7aa648', 'c': '#c0392b', 'cur': '#3a6ea5', 'faat': '#c0392b'}
ARM_LABEL = {'a': 'arm a (target-only)', 'b': 'arm b (+OOD data)', 'c': 'arm c (OOD calib.)',
             'cur': 'arm cur (universal)', 'faat': 'FAAT (frozen recipe)'}


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(r, k):
    return float(r[k])


def main():
    R = rows_of(HERE.parent / 'docs' / 'pilot_metrics.csv')
    F = rows_of(HERE.parent / 'docs' / 'pilot_faat_family.csv')

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))

    # ---------------- Panel A: budget axis (GTSRB, main + pr sweep) ----------
    ax = axes[0]
    for arm in ['a', 'cur']:
        pts = {}
        for r in R:
            if r['dataset'] != 'gtsrb' or r['arm'] != arm:
                continue
            m = re.search(r'_pr(0\.\d+)', r['tag'])
            pr = float(m.group(1)) if m else 0.01
            pts.setdefault(pr, []).append(f(r, 'victim_asr'))
        xs = sorted(pts)
        ax.errorbar(xs, [np.mean(pts[x]) for x in xs],
                    yerr=[np.std(pts[x]) for x in xs], marker='o', ms=3,
                    lw=1.2, capsize=2,
                    color={'a': '#e69138', 'cur': '#3a6ea5'}[arm], label=ARM_LABEL[arm])
    ax.set_xscale('log')
    ax.set_xlabel('poisoning rate')
    ax.set_ylabel('victim ASR (%)')
    ax.set_title('(a) budget axis: flat', fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc='center right')

    # ---------------- Panel B: calibration axis (GTSRB arm c, w sweep) -------
    # w=0 is defined as arm c with calibration off == arm a (HANDOFF: 10/10 acquired);
    # untagged c runs ran at the default w=1.0.
    ax = axes[1]
    pts = {}
    a_vals = [f(r, 'victim_asr') for r in R
              if r['dataset'] == 'gtsrb' and r['arm'] == 'a'
              and '_pr0.' not in r['tag'] and '_ep150' not in r['tag']]
    pts[0.0] = a_vals
    for r in R:
        if r['dataset'] != 'gtsrb' or r['arm'] != 'c':
            continue
        m = re.search(r'_w(0\.\d+)', r['tag'])
        if not m:
            continue                      # untagged c runs are w=1.0
        w = float(m.group(1))
        pts.setdefault(w, []).append(f(r, 'victim_asr'))
    pts.setdefault(1.0, [])
    for r in R:
        if r['dataset'] == 'gtsrb' and r['arm'] == 'c' and '_w' not in r['tag'] \
                and '_ep150' not in r['tag'] and '_pr0.' not in r['tag']:
            pts[1.0].append(f(r, 'victim_asr'))
    xs = sorted(pts)
    ax.errorbar(xs, [np.mean(pts[x]) for x in xs],
                yerr=[np.std(pts[x]) if len(pts[x]) > 1 else 0 for x in xs],
                marker='s', ms=3.5, lw=1.2, capsize=2, color=ARM_COLOR['c'])
    ax.axhspan(0, 20, color='#c0392b', alpha=0.08)
    ax.text(0.3, 9, 'non-acquisition\nregime', fontsize=6, color='#c0392b')
    ax.set_xlabel('OOD weight $w$   ($w{=}0 \\equiv$ arm a)')
    ax.set_ylabel('victim ASR (%)')
    ax.set_title('(b) calibration axis: cliff at $w{\\approx}0.05$', fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # ---------------- Panel C: generality -> ASR (incl. FAAT family) ---------
    ax = axes[2]
    ds_marker = {'gtsrb': 'o', 'cifar10': 's', 'cifar100': '^'}
    for r in R:
        if '_ep150' in r['tag'] or '_pr0.' in r['tag'] or '_w0.' in r['tag']:
            continue
        ax.scatter(f(r, 'proxy_gen_nontgt'), f(r, 'victim_asr'),
                   marker=ds_marker[r['dataset']], s=14, alpha=0.75,
                   color=ARM_COLOR.get(r['arm'], 'gray'),
                   edgecolors='none')
    for r in F:
        ax.scatter(f(r, 'proxy_gen_nontgt'), f(r, 'test_triggered_asr'),
                   marker='*', s=42, color='#c0392b', edgecolors='k',
                   linewidths=0.3, zorder=5)
    # rho (Spearman), two calibers: pooled main-grid+FAAT, and GTSRB-with-sweeps
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]

    gx = [f(r, 'proxy_gen_nontgt') for r in R if not any(k in r['tag'] for k in ('_ep150', '_pr0.', '_w0.'))]
    gy = [f(r, 'victim_asr') for r in R if not any(k in r['tag'] for k in ('_ep150', '_pr0.', '_w0.'))]
    gx += [f(r, 'proxy_gen_nontgt') for r in F]
    gy += [f(r, 'test_triggered_asr') for r in F]
    rho = spearman(gx, gy)
    g_all = [r for r in R if r['dataset'] == 'gtsrb' and '_ep150' not in r['tag']]
    rho_g = spearman([f(r, 'proxy_gen_nontgt') for r in g_all],
                     [f(r, 'victim_asr') for r in g_all])
    ax.scatter([], [], marker='*', s=42, color='#c0392b', edgecolors='k',
               linewidths=0.3,
               label=f'FAAT family (pooled $\\rho_s$={rho:.2f}; GTSRB {rho_g:.2f})')
    ax.set_xlabel('proxy generality $g_{nt}$')
    ax.set_ylabel('victim ASR (%)')
    ax.set_title('(c) generality predicts ASR', fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=6, loc='lower right')

    # ---------------- Panel D: detection decouples from ASR ------------------
    ax = axes[3]
    # AC-AUC per dataset (mean over main-grid runs) vs mean victim ASR
    acs = {'gtsrb': 0.77, 'cifar10': 0.25, 'cifar100': 0.02}   # HANDOFF RQ4 values
    means = {}
    for ds in acs:
        vals = [f(r, 'victim_asr') for r in R
                if r['dataset'] == ds and all(k not in r['tag'] for k in ('_ep150', '_pr0.', '_w0.'))]
        means[ds] = np.mean(vals)
    order = ['gtsrb', 'cifar10', 'cifar100']
    ax.plot([means[d] for d in order], [acs[d] for d in order],
            marker='o', ms=4, lw=1.2, color='#3a6ea5')
    for d in order:
        ax.annotate(d, (means[d], acs[d]), textcoords='offset points',
                    xytext=(4, 4), fontsize=6)
    ax.set_xlabel('mean victim ASR (%)')
    ax.set_ylabel('AC-AUC (poison detectability)')
    ax.set_title('(d) more learnable, less detectable', fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / 'fig_phase_diagram.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig_phase_diagram.png', dpi=200, bbox_inches='tight')
    print(f'wrote {OUT}/fig_phase_diagram.pdf (rho={rho:.3f})')


if __name__ == '__main__':
    main()
