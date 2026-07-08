#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_ablation.py — Ablation study figures for backdoor attack papers.

Usage:
  python plot_ablation.py ablation.csv -o output/figs/
  python plot_ablation.py ablation.csv --full-name "Full Method" --metrics ASR BA
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_parser, academic_caption,
    FALLBACK_PALETTE, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')


def load_ablation_csv(csv_path):
    """Load ablation results, auto-detecting format."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Check if long format or wide format
    if 'variant' in df.columns and 'metric' in df.columns and 'value' in df.columns:
        # Long format — good
        return df, 'long'

    # Wide format: variant names as columns
    potential_variants = [c for c in df.columns if c not in ('dataset', 'poison_rate', 'metric')]
    if len(potential_variants) > 1:
        id_cols = [c for c in df.columns if c in ('dataset', 'poison_rate', 'metric')]
        df_long = df.melt(id_vars=id_cols, value_vars=potential_variants,
                          var_name='variant', value_name='value')
        return df_long, 'long'

    return df, 'unknown'


def plot_ablation_bars(df, args):
    """Plot ablation study as horizontal bar chart."""
    full_name = getattr(args, 'full_name', 'Full Method')
    metrics = getattr(args, 'metrics', None) or ['ASR', 'BA']

    available_metrics = [m for m in metrics if m in df['metric'].values]
    if not available_metrics:
        available_metrics = df['metric'].unique().tolist()

    # Aggregate mean per variant per metric
    agg = df.groupby(['variant', 'metric'])['value'].mean().reset_index()

    # Sort: Full Method first, then by ASR descending
    variants = agg['variant'].unique()
    full_variants = [v for v in variants if full_name.lower() in str(v).lower()]
    other_variants = [v for v in variants if full_name.lower() not in str(v).lower()]

    # Sort others by ASR (descending)
    asr_for_sort = agg[(agg['metric'] == 'ASR') & (agg['variant'].isin(other_variants))]
    variant_asr = dict(zip(asr_for_sort['variant'], asr_for_sort['value']))
    other_variants = sorted(other_variants, key=lambda v: variant_asr.get(v, 0), reverse=True)

    ordered_variants = full_variants + other_variants
    n_variants = len(ordered_variants)

    fig, axes = plt.subplots(1, len(available_metrics),
                             figsize=args.figsize or DOUBLE_COLUMN)
    if len(available_metrics) == 1:
        axes = [axes]

    for ax_idx, metric in enumerate(available_metrics):
        ax = axes[ax_idx]

        values = []
        for v in ordered_variants:
            row = agg[(agg['variant'] == v) & (agg['metric'] == metric)]
            values.append(row['value'].values[0] if not row.empty else 0)

        # Horizontal bars
        y_pos = range(n_variants)
        colors = []
        for i, v in enumerate(ordered_variants):
            if full_name.lower() in str(v).lower():
                colors.append('#c0392b')  # Full method in bold red
            else:
                colors.append(FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])

        bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(v) for v in ordered_variants], fontsize=8)
        ax.set_xlabel(f'{metric} (%)')
        ax.set_title(f'{metric}' if metric != 'ASR' else 'Attack Success Rate')
        ax.invert_yaxis()  # Full method at top

        # Annotate values and Δ from full method
        full_value = values[0] if full_variants else values[0]
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=8)
                if i > 0 and full_value > 0 and metric == 'ASR':
                    delta = val - full_value
                    delta_text = f'  Δ={delta:+.1f}'
                    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                            delta_text, va='center', fontsize=7,
                            color='red' if delta < -1 else 'green' if delta > 1 else 'gray')

        # Set x-limit with room for annotations
        ax.set_xlim(0, max(values) * 1.2 if values else 100)

    fig.suptitle('Ablation Study — Component Contribution',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        if len(ordered_variants) > 1 and values:
            max_drop = min(values[1:]) - values[0] if values[0] > 0 else 0
            max_drop_variant = ordered_variants[1:][np.argmin(values[1:])] if len(values) > 1 else ''
            print(academic_caption('ablation',
                  dataset='(provided)',
                  ablation_target=str(max_drop_variant),
                  asr_drop=f'{-max_drop:.1f}'))

    return fig


def main():
    parser = get_parser("Ablation study plots for backdoor papers.")
    parser.add_argument('--full-name', default='Full Method',
                        help='Name of the full method variant (default: "Full Method")')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to plot (default: ASR BA)')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    csv_path = args.input
    if csv_path is None:
        parser.error("Input CSV file required")

    print(f"Loading: {csv_path}")
    df, fmt = load_ablation_csv(csv_path)
    print(f"  Format: {fmt}, Variants: {df['variant'].nunique() if 'variant' in df.columns else 'N/A'}")
    if 'metric' in df.columns:
        print(f"  Metrics: {df['metric'].unique().tolist()}")

    plot_ablation_bars(df, args)

    if args.metadata:
        meta = {
            'input': csv_path,
            'format': fmt,
            'full_method_name': args.full_name,
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    main()
