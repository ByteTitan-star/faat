#!/usr/bin/env python
"""Generate paper figures (Fig 1-5) into docs/figs/. No GPU needed.
Data: resource/kst_sdt/*.pt (spectra), results/kst_sdt/.../output_*.log (curves),
paper_tables.md (Pareto/confidence numbers, hardcoded below)."""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

OUT = 'docs/figs'
os.makedirs(OUT, exist_ok=True)


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(8.5, 3.2)); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(txt, x, y, w=0.20, h=0.26, fc='white'):
        ax.add_patch(plt.Rectangle((x, y - h/2), w, h, fill=True, facecolor=fc, edgecolor='k', lw=1.5))
        ax.text(x + w/2, y, txt, ha='center', va='center', fontsize=9)
    box(r'$x$' + '\n(target class)', 0.04, 0.5, fc='#eef')
    box(r'$\tilde{x}=x+\delta_g+\delta_a$', 0.34, 0.5, w=0.24, fc='#efe')
    box('Victim\nResNet-18', 0.72, 0.5, fc='#fee')
    for x0, x1 in [(0.24, 0.34), (0.58, 0.72), (0.92, 0.99)]:
        ax.annotate('', xy=(x1, 0.5), xytext=(x0, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(0.46, 0.78, r'$\delta_g$ frozen Narcissus global', fontsize=8, ha='center')
    ax.text(0.46, 0.70, r'$\delta_a$ bounded DCT adaptive', fontsize=8, ha='center')
    ax.annotate('', xy=(0.46, 0.63), xytext=(0.46, 0.72), arrowprops=dict(arrowstyle='->', lw=1))
    ax.text(0.82, 0.20, r'$\mathcal{L}_{align}$ pulls poisoned\nfeature → target centroid',
            fontsize=8, ha='center', color='#a30')
    ax.set_title('Figure 1 — FAAT pipeline')
    plt.savefig(f'{OUT}/fig1_pipeline.png'); plt.close()


def fig2_spectrum():
    try:
        import torch
        kst = torch.load('resource/kst_sdt/kst_delta_r4_eps0.0627.pt', map_location='cpu')
        narc = torch.load('resource/kst_sdt/narc_delta_eps0.0627.pt', map_location='cpu')
        kst = kst.numpy().astype(float) if hasattr(kst, 'numpy') else np.asarray(kst, float)
        narc = narc.numpy().astype(float) if hasattr(narc, 'numpy') else np.asarray(narc, float)
    except Exception as e:
        print(f"  fig2 skip ({e})"); return
    def radial_ps(d):
        d = d.reshape(-1, *d.shape[-2:]) if d.ndim == 2 else d
        F = np.fft.fftshift(np.abs(np.fft.fft2(d, axes=(-2, -1))).mean(axis=0))
        H, W = F.shape; yy, xx = np.indices((H, W)); cy, cx = H//2, W//2
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2).astype(int)
        n = np.bincount(r.ravel())
        ps = np.bincount(r.ravel(), F.ravel()) / np.maximum(n, 1)
        return ps[1:] / max(ps[1:].mean(), 1e-12)
    pk = radial_ps(kst); pn = radial_ps(narc)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(np.arange(1, len(pk)+1), pk, 'b-', lw=2, label=f'KST (flat, peak={pk.max():.1f})')
    ax.semilogy(np.arange(1, len(pn)+1), pn, 'r-', lw=2, label=f'Narcissus (peak={pn.max():.1f})')
    ax.set_xlabel('radial frequency'); ax.set_ylabel('normalized power')
    ax.legend(fontsize=9); ax.set_title(r'Figure 2 — Trigger amplitude spectrum (CIFAR-10, $\epsilon$=16/255)')
    plt.savefig(f'{OUT}/fig2_spectrum.png'); plt.close()


def fig3_pareto():
    data = {
        'CIFAR-10':  {'FAAT': (0.950, 93.6), 'KST': (0.94, 92.6)},
        'CIFAR-100': {'FAAT': (0.919, 99.6), 'KST': (0.94, 94.0)},
        'Tiny':      {'FAAT': (0.927, 95.8), 'KST': (0.94, 98.5)},
        'GTSRB':     {'FAAT': (0.720, 82.7), 'KST': (0.94, 0.8)},
    }
    colors = {'CIFAR-10': 'C0', 'CIFAR-100': 'C1', 'Tiny': 'C2', 'GTSRB': 'C3'}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for ds, d in data.items():
        for m, (ssim, asr) in d.items():
            ax.scatter(ssim, asr, c=colors[ds], marker=('o' if m == 'FAAT' else 's'),
                       s=90, edgecolor='k', zorder=3)
            ax.annotate(m, (ssim, asr), fontsize=7, xytext=(5, 4), textcoords='offset points')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='FAAT', markersize=10),
               Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='KST', markersize=10)]
    for ds, c in colors.items():
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=ds, markersize=10))
    ax.legend(handles=handles, fontsize=8, loc='lower left')
    ax.set_xlabel('Stealth (SSIM ↑)'); ax.set_ylabel('Attack (ASR ↑)')
    ax.set_title('Figure 3 — ASR vs stealth across datasets')
    plt.savefig(f'{OUT}/fig3_pareto.png'); plt.close()


def fig4_confidence():
    ba = [55, 78, 95, 100]; kst = [98.5, 94.0, 92.6, 0.8]; faat = [95.8, 99.6, 93.6, 82.7]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ba, kst, 'rs--', label='KST (flat-spectrum)', markersize=11, lw=2)
    ax.plot(ba, faat, 'bo-', label='FAAT (optimized)', markersize=11, lw=2)
    for x, y in zip(ba, kst):   ax.annotate(f'{y}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
    for x, y in zip(ba, faat):  ax.annotate(f'{y}', (x, y), xytext=(5, -12), textcoords='offset points', fontsize=8, color='blue')
    ax.axvspan(98, 101, alpha=0.12, color='gray'); ax.text(99.2, 88, 'GTSRB\nPareto\nswap', fontsize=8, ha='center')
    ax.set_xlabel('victim clean accuracy (%)  ←  confidence'); ax.set_ylabel('ASR (%)')
    ax.legend(fontsize=9); ax.set_title('Figure 4 — Behavior along victim confidence')
    plt.savefig(f'{OUT}/fig4_confidence.png'); plt.close()


def fig5_gtsrb_curve():
    gs = sorted(glob.glob('results/kst_sdt/gtsrb_kst_e48_forget/output_*.log'))
    if not gs: print("  fig5 skip (no log)"); return
    eps, asrs = [], []
    for line in open(gs[0]):
        m = re.match(r'^\[[^\]]+\]\s*-\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line.strip())
        if m: eps.append(int(m[1])); asrs.append(100*float(m.group(7)))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(eps, asrs, 'b-', lw=1.4)
    pi = int(np.argmax(asrs))
    ax.annotate(f'peak {asrs[pi]:.1f}% @ ep{eps[pi]}', (eps[pi], asrs[pi]),
                xytext=(eps[pi]+30, asrs[pi]+8), arrowprops=dict(arrowstyle='->'))
    ax.axhline(0.8, ls='--', color='gray'); ax.text(5, 3, 'terminal 0.8%', color='gray', fontsize=8)
    ax.set_xlabel('epoch'); ax.set_ylabel('ASR (%)')
    ax.set_title('Figure 5 — KST ε48 on GTSRB: learns briefly, then victim resists')
    plt.savefig(f'{OUT}/fig5_gtsrb_curve.png'); plt.close()


if __name__ == '__main__':
    for f in [fig1_pipeline, fig2_spectrum, fig3_pareto, fig4_confidence, fig5_gtsrb_curve]:
        try: f(); print(f"  {f.__name__} ok")
        except Exception as e: print(f"  {f.__name__} FAIL: {e}")
    print("Done. Figures in", OUT)
