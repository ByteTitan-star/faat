#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_backdoor_metrics.py — Paper-ready ASR/ACC/BA metric plots.

Usage:
  python plot_backdoor_metrics.py docs/v6c_results.csv -o output/figs/
  python plot_backdoor_metrics.py docs/v6c_results.csv --highlight FAAT --figsize 7 3
  python plot_backdoor_metrics.py docs/gtsrb_baselines_results.csv --bar --prefix gtsrb

Supports:
  - Wide CSV (project native): run,dataset,L2,seed,ASR,BA,...
  - Long CSV (generic):     dataset,attack,defense,poison_rate,metric,value
  - GTSRB baseline CSV:     run,attack,seed,ASR,BA
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_color, get_marker,
    get_parser, load_wide_csv, academic_caption,
    METHOD_COLORS, MARKERS, FALLBACK_PALETTE,
    SINGLE_COLUMN, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_and_prepare(csv_path):
    """Load CSV and prepare standardized data for plotting."""
    # Try loading as the project's wide format first
    df = load_wide_csv(csv_path)

    # Check if it's already long format (has metric/value columns)
    if 'metric' in df.columns and 'value' in df.columns:
        return df, 'long'

    # Check if it has id_cols from wide format conversion
    id_cols = df.attrs.get('id_cols', [])
    metric_cols = df.attrs.get('metric_cols', [])

    if id_cols and metric_cols:
        # Already melted to long
        return df, 'long'

    # Try to detect format from column names
    cols = set(df.columns)
    run_cols = {'run', 'dataset', 'attack', 'seed', 'L2', 'poison_rate'}
    metric_cols_found = [c for c in ['ASR', 'BA', 'ACC', 'RA'] if c in cols]

    if metric_cols_found and (run_cols & cols):
        # Wide format — melt
        id_cols = [c for c in df.columns if c not in metric_cols_found]
        df_long = df.melt(id_vars=id_cols, value_vars=metric_cols_found,
                          var_name='metric', value_name='value')
        df_long.attrs['wide'] = df
        return df_long, 'long'

    return df, 'unknown'


def get_dataset_name(df_long):
    """Extract dataset name from DataFrame."""
    if 'dataset' in df_long.columns:
        vals = df_long['dataset'].dropna().unique()
        if len(vals) > 0:
            name = str(vals[0])
            # Normalize common names
            name_map = {
                'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
                'gtsrb': 'GTSRB', 'tinyimagenet': 'Tiny-ImageNet',
                'imagenet': 'ImageNet',
            }
            return name_map.get(name.lower(), name.upper())
    return 'Unknown'


def get_method_names(df_long):
    """Extract method/attack names from DataFrame."""
    for col in ['attack', 'run', 'method']:
        if col in df_long.columns:
            names = df_long[col].dropna().unique()
            # Try to extract method from run name pattern "method_..."
            patterns = []
            for n in names:
                n = str(n)
                # "cifar100_l2_1.5_seed1" → "FAAT"
                # "badnets_seed1" → "BadNets"
                if n.startswith('cifar') or n.startswith('gtsrb') or 'l2' in n:
                    patterns.append('FAAT')
                else:
                    parts = n.replace('_seed', '|seed').split('|')[0]
                    patterns.append(parts)
            return sorted(set(patterns), key=lambda x: list(METHOD_COLORS.keys()).index(x) if x in METHOD_COLORS else 999)
    return []


def plot_asr_vs_param(df_long, x_col, metrics, output_dir, prefix, args):
    """Plot ASR and BA as line plots against a continuous parameter (e.g., L2, poison_rate)."""
    fig, axes = plt.subplots(1, 2, figsize=args.figsize or DOUBLE_COLUMN)

    # Determine grouping
    group_col = None
    for col in ['attack', 'run', 'dataset']:
        if col in df_long.columns and df_long[col].nunique() > 1:
            group_col = col
            break

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        metric_data = df_long[df_long['metric'] == metric].copy()
        if metric_data.empty:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            continue

        if group_col and group_col in metric_data.columns:
            groups = metric_data[group_col].unique()
            for i, grp in enumerate(groups):
                grp_data = metric_data[metric_data[group_col] == grp].sort_values(x_col)
                label = str(grp)
                color = get_color(label) or FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)]
                marker = get_marker(i)
                ax.plot(grp_data[x_col], grp_data['value'], marker=marker, color=color,
                        label=label, linewidth=1.5, markersize=5, markeredgewidth=0.5,
                        markeredgecolor='white')
                # Add seed-wise scatter if multiple seeds
                if 'seed' in metric_data.columns and metric_data['seed'].nunique() > 1:
                    seed_data = grp_data.groupby(x_col)['value'].agg(['mean', 'std']).reset_index()
                    ax.fill_between(seed_data[x_col],
                                    seed_data['mean'] - seed_data['std'],
                                    seed_data['mean'] + seed_data['std'],
                                    alpha=0.15, color=color)
        else:
            agg = metric_data.groupby(x_col)['value'].agg(['mean', 'std']).reset_index()
            ax.errorbar(agg[x_col], agg['mean'], yerr=agg['std'],
                        fmt='o-', color='#c0392b', capsize=3, linewidth=1.5, markersize=5)

        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(f'{metric}' if metric != 'ASR' else 'Attack Success Rate')
        if group_col:
            ax.legend(fontsize=8, ncol=1)

        # ASR specific: annotate "higher is better"
        if metric in ('ASR',):
            ax.annotate('↑ better', xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=7, color='gray',
                        fontstyle='italic')

    fig.suptitle(f'{get_dataset_name(df_long)} — Attack Performance',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(output_dir) / prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        asr_max = df_long[(df_long['metric']=='ASR')]['value'].max()
        ba_best = df_long[(df_long['metric'].isin(['BA','ACC']))]['value'].max()
        print(academic_caption('metric_comparison',
              dataset=get_dataset_name(df_long), methods=' and '.join(get_method_names(df_long)),
              ours='FAAT', best_asr=f'{asr_max:.1f}', best_ba=f'{ba_best:.1f}'))

    return fig


def plot_attack_comparison_bars(df_long, output_dir, prefix, args):
    """Plot grouped bar chart comparing attack methods."""
    metrics = ['ASR', 'BA'] if 'BA' in df_long['metric'].values else ['ASR', 'ACC']
    available = [m for m in metrics if m in df_long['metric'].values]

    method_col = None
    for col in ['attack', 'run']:
        if col in df_long.columns:
            method_col = col
            break

    if method_col is None:
        print("Warning: no method column found; falling back to line plot")
        return plot_asr_vs_param(df_long, list(df_long.columns)[0], available, output_dir, prefix, args)

    # Aggregate: mean per method per metric
    agg = df_long.groupby([method_col, 'metric'])['value'].agg(['mean', 'std']).reset_index()

    methods = sorted(agg[method_col].unique(), key=lambda x: str(x))
    n_methods = len(methods)
    n_metrics = len(available)

    fig, axes = plt.subplots(1, n_metrics, figsize=args.figsize or DOUBLE_COLUMN)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    width = 0.55

    for ax_idx, metric in enumerate(available):
        ax = axes[ax_idx]
        values = []
        errors = []
        for method in methods:
            row = agg[(agg[method_col] == method) & (agg['metric'] == metric)]
            if not row.empty:
                values.append(row['mean'].values[0])
                errors.append(row['std'].values[0] if 'std' in row else 0)
            else:
                values.append(0)
                errors.append(0)

        bars = ax.bar(x, values, width, color=[get_color(str(m)) or FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)] for i, m in enumerate(methods)],
                      edgecolor='white', linewidth=0.5)
        if any(e > 0 for e in errors):
            ax.errorbar(x, values, yerr=errors, fmt='none', ecolor='#333333', capsize=3, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in methods], rotation=15, ha='right', fontsize=8)
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(metric if metric == 'ASR' else 'Benign Accuracy')

        # Annotate values on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        # Set y-limit with some headroom
        ymax = max(values) * 1.12 if values else 100
        ax.set_ylim(0, min(ymax, 105))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    fig.suptitle(f'{get_dataset_name(df_long)} — Method Comparison',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(output_dir) / prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        print(academic_caption('metric_comparison',
              dataset=get_dataset_name(df_long), methods=str(list(methods)),
              ours='FAAT', best_asr=f'{max(values):.1f}', best_ba='N/A'))

    return fig


def main():
    parser = get_parser("Plot backdoor attack metrics for research papers.")
    parser.add_argument('--bar', action='store_true',
                        help='Force grouped bar chart (vs. auto-detect)')
    parser.add_argument('--highlight', default=None,
                        help='Method name to highlight as "ours" (e.g., FAAT)')
    parser.add_argument('--x', default=None,
                        help='Column to use as x-axis (auto-detect if not specified)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to plot (default: ASR BA)')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    csv_path = args.input
    if csv_path is None:
        # Try to find project CSVs
        from backdoor_paper_style import find_project_csv
        found = find_project_csv()
        if found:
            csv_path = str(found[0])
            print(f"Auto-detected input: {csv_path}")
        else:
            parser.error("No input CSV specified and none found in docs/")

    print(f"Loading: {csv_path}")
    df_long, fmt = load_and_prepare(csv_path)
    print(f"  Detected format: {fmt}")
    print(f"  Rows: {len(df_long)}, Metrics: {df_long['metric'].unique().tolist() if 'metric' in df_long.columns else 'N/A'}")

    # Determine what to plot
    metrics_to_plot = args.metrics or ['ASR', 'BA']
    available = [m for m in metrics_to_plot if m in df_long['metric'].values]
    if not available:
        print(f"Warning: none of {metrics_to_plot} found; using all metrics")
        available = df_long['metric'].unique().tolist()

    # Determine x-axis
    x_col = args.x
    if x_col is None:
        for candidate in ['L2', 'poison_rate', 'trigger_size', 'alpha', 'strength',
                          'defense', 'attack', 'run', 'dataset']:
            if candidate in df_long.columns and df_long[candidate].nunique() > 1:
                x_col = candidate
                break

    # Decide plot type
    if args.bar or (x_col and df_long[x_col].dtype == 'object' and df_long[x_col].nunique() <= 10):
        plot_attack_comparison_bars(df_long, args.output, args.prefix, args)
    else:
        plot_asr_vs_param(df_long, x_col, available, args.output, args.prefix, args)

    # Save metadata
    if args.metadata:
        meta = {
            'input': csv_path,
            'format': fmt,
            'metrics': available,
            'x_column': x_col,
            'figure_size': args.figsize or list(DOUBLE_COLUMN),
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    main()
