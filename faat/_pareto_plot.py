"""Headline Pareto figure for the saturation analysis: ASR vs SSIM across trigger mechanisms.

Narcissus / ICIT / BppAttack cluster at the front (high ASR + high SSIM); the 6 failed
mechanisms sit below (either high-ASR-but-visible, or stealthy-but-weak). Visual proof of
Pareto saturation.

Run: python -m faat._pareto_plot
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, ASR, SSIM, type, marker)  -- ASR is victim-ASR for front, proxyASR(*) for failed
# (*proxyASR is the FAVORABLE bar: if it can't flip a clean proxy, it can't work as a victim trigger.)
PTS = [
    # front (victim ASR)
    ('Narcissus',       0.88, 0.95, 'front', 's'),
    ('ICIT',            0.997, 0.94, 'front', 's'),
    ('BppAttack',       0.90, 0.97, 'front', 's'),
    ('BadNets-C',       0.71, 0.92, 'visible', '^'),
    ('Blended-C',       0.73, 0.92, 'visible', '^'),
    # failed optimized-type (proxyASR)
    ('RKT (best)',      0.82, 0.48, 'failed', 'X'),
    ('RKT (stealthy)',  0.07, 0.62, 'failed', 'X'),
    ('dither (best)',   0.54, 0.81, 'failed', 'X'),
    ('dither (stealthy)', 0.00, 0.92, 'failed', 'X'),
    ('Chroma',          0.045, 0.95, 'failed', 'X'),
    ('freq-band B8',    0.46, 0.60, 'failed', 'X'),
    ('nat-texture',     0.002, 0.95, 'failed', 'X'),
]

COLOR = {'front': '#1b7837', 'visible': '#762a83', 'failed': '#c51b7d'}


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, asr, ssim, kind, m in PTS:
        ax.scatter([ssim], [asr], s=120, marker=m, c=COLOR[kind], edgecolors='k',
                   linewidths=0.6, zorder=3, label=kind)
        # annotate
        dy = 0.02 if not label.startswith('RKT (best') else -0.05
        ax.annotate(label, (ssim, asr), textcoords='offset points', xytext=(6, 6),
                    fontsize=8)
    # dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9, framealpha=0.9)
    # shade the "saturated front" region (SSIM>=0.9, ASR>=0.8) -- no failed point enters
    ax.axvspan(0.9, 1.0, ymin=0.0, ymax=1.0, alpha=0.08, color='green', zorder=0)
    ax.axhline(0.8, ls='--', lw=0.8, color='gray', alpha=0.6)
    ax.axvline(0.9, ls='--', lw=0.8, color='gray', alpha=0.6)
    ax.text(0.945, 0.15, 'saturated front\n(no failed mechanism enters)',
            fontsize=8, color='#1b7837', ha='center')
    ax.set_xlabel('Stealth  (SSIM ↑)', fontsize=11)
    ax.set_ylabel('Effectiveness  (ASR ↑)', fontsize=11)
    ax.set_title('Clean-label backdoor trigger Pareto front is saturated (CIFAR-10)', fontsize=11)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    os.makedirs('docs/figs', exist_ok=True)
    out = 'docs/figs/pareto_saturation.png'
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print('saved', out)


if __name__ == '__main__':
    main()
