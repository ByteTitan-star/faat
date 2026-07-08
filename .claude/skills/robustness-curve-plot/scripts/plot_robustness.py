#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_robustness.py — Robustness curve plots for backdoor attack papers.

Usage:
  python plot_robustness.py robustness.csv -o output/figs/
  python plot_robustness.py robustness.csv --threshold 80 -o output/figs/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_color, get_marker,
    get_parser, academic_caption,
    METHOD_COLORS, MARKERS, FALLBACK_PALETTE, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')


def load_robustness_csv(csv_path):
    """Load robustness results CSV."""
    df = pd.read_csv(csv_path)
    # Normalize
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Ensure numeric
    for col in ['strength', 'value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_robustness_curves(df, args):
    """Plot ASR vs transformation strength curves."""
    transformations = df['transformation'].unique()
    n_trans = len(transformations)

    # Decide layout: grid of subplots
    n_cols = min(3, n_trans)
    n_rows = int(np.ceil(n_trans / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=args.figsize or (n_cols * 2.8, n_rows * 2.3))
    if n_trans == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Check for attack grouping
    attacks = df['attack'].unique() if 'attack' in df.columns else [None]
    show_legend_per_subplot = len(attacks) > 1

    for idx, trans in enumerate(transformations):
        ax = axes_flat[idx]
        trans_data = df[df['transformation'] == trans].sort_values('strength')

        for att_idx, attack in enumerate(attacks):
            if attack is not None:
                att_data = trans_data[trans_data['attack'] == attack]
            else:
                att_data = trans_data

            if att_data.empty:
                continue

            label = str(attack) if attack else trans
            color = get_color(str(attack)) if attack else FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]
            marker = get_marker(att_idx)

            ax.plot(att_data['strength'], att_data['value'],
                    marker=marker, color=color, label=label if show_legend_per_subplot else None,
                    linewidth=1.5, markersize=4, markeredgewidth=0.3, markeredgecolor='white')

        # Add threshold line
        threshold = getattr(args, 'threshold', 80)
        if threshold:
            ax.axhline(y=threshold, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.6)
            # Find where ASR drops below threshold
            asr_data = trans_data[trans_data['metric'] == 'ASR'] if 'metric' in trans_data.columns else trans_data
            if not asr_data.empty:
                below = asr_data[asr_data['value'] < threshold]
                if not below.empty:
                    critical_strength = below.iloc[0]['strength']
                    ax.annotate(f'Below {threshold}% at {critical_strength}',
                                xy=(critical_strength, threshold),
                                xytext=(10, -15), textcoords='offset points',
                                fontsize=6, color='#e74c3c',
                                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8))

        # Clean ASR baseline
        if 'ASR' in asr_data['metric'].values if 'metric' in asr_data.columns else True:
            asr_for_baseline = asr_data[asr_data['metric'] == 'ASR']['value'].max() if 'metric' in asr_data.columns else asr_data['value'].max()
            ax.axhline(y=asr_for_baseline, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)

        ax.set_title(trans.replace('_', ' ').title(), fontsize=9)
        ax.set_xlabel('Strength')
        ax.set_ylabel('ASR (%)')
        ax.set_ylim(0, 105)

        if show_legend_per_subplot and len(attacks) <= 4:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_trans, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Attack Robustness Under Transformations', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        n_transforms = len(transformations)
        print(academic_caption('robustness',
              dataset='(provided)', transformations=', '.join(transformations),
              asr_threshold=f'{threshold}%' if threshold else 'N/A',
              n_transforms=n_transforms))

    return fig


def main():
    parser = get_parser("Robustness curve plots for backdoor papers.")
    parser.add_argument('--threshold', type=float, default=80,
                        help='ASR threshold line (default: 80%%)')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    csv_path = args.input
    if csv_path is None:
        parser.error("Input CSV file required")

    print(f"Loading: {csv_path}")
    df = load_robustness_csv(csv_path)
    print(f"  Rows: {len(df)}, Transformations: {df['transformation'].nunique()}")
    if 'attack' in df.columns:
        print(f"  Attacks: {df['attack'].nunique()}")

    plot_robustness_curves(df, args)

    if args.metadata:
        meta = {
            'input': csv_path,
            'transformations': df['transformation'].unique().tolist(),
            'threshold': args.threshold,
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    main()
