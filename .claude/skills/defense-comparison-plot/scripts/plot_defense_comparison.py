#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_defense_comparison.py — Backdoor defense evaluation figures.

Usage:
  python plot_defense_comparison.py defense_results/summary.csv -o output/figs/
  python plot_defense_comparison.py defense_results/summary.csv --scenario badnet_C_p0.03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_color, get_marker,
    get_parser, load_defense_csv, academic_caption,
    METHOD_COLORS, MARKERS, FALLBACK_PALETTE,
    DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def prepare_defense_data(df, scenario=None, selection=None):
    """Prepare defense results for plotting.

    Parameters
    ----------
    df : DataFrame
        Loaded defense results
    scenario : str, optional
        Filter to specific scenario (e.g., 'badnet_C_p0.03')
    selection : str, optional
        Filter to specific selection strategy (e.g., 'forget')

    Returns
    -------
    DataFrame with columns: defense, category, test_acc, test_asr
    """
    df = df.copy()

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(' ', '_')
        col_map[c] = cl
    df = df.rename(columns=col_map)

    # Filter
    if scenario and 'scenario' in df.columns:
        df = df[df['scenario'] == scenario]
    if selection and 'selection' in df.columns:
        df = df[df['selection'] == selection]

    # Aggregate (mean per defense, in case multiple runs)
    agg_cols = {'test_acc': 'mean', 'test_asr': 'mean'}
    group_cols = ['defense', 'category'] if 'category' in df.columns else ['defense']
    # Also include TPR/FPR if present
    for c in ['tpr', 'fpr']:
        if c in df.columns:
            agg_cols[c] = 'mean'

    if group_cols[0] in df.columns:
        agg = df.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    else:
        agg = df

    return agg


def plot_defense_asr_acc(agg, args):
    """Two-panel: ASR (left) and ACC (right) after each defense."""
    defenses = agg['defense'].unique()

    # Filter out 'None' / 'No Defense' for sorting, but keep for reference
    defense_order = sorted(
        [d for d in defenses if str(d).lower() not in ('none', 'no defense', '')],
        key=lambda d: agg[agg['defense'] == d]['test_asr'].mean()
        if 'test_asr' in agg.columns else 0
    )

    fig, axes = plt.subplots(1, 2, figsize=args.figsize or DOUBLE_COLUMN)

    n = len(defense_order)
    x = np.arange(n)
    width = 0.55

    defense_colors = {
        'abl': '#1b9e77', 'nc': '#d95f02', 'fp': '#1b9e77',
        'fst': '#666666', 'rnp': '#999999', 'ac': '#8c564b',
        'i-bau': '#a6761d', 'anp': '#66a61e', 'nad': '#e7298a',
        'strip': '#7570b3',
    }

    # Panel 1: ASR after defense
    ax = axes[0]
    asr_values = []
    for d in defense_order:
        row = agg[agg['defense'] == d]
        asr_values.append(row['test_asr'].values[0] * 100 if 'test_asr' in row else 0)
    colors = [defense_colors.get(str(d).lower().replace('_', '-'),
                                 FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])
              for i, d in enumerate(defense_order)]
    ax.bar(x, asr_values, width, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d).upper().replace('_', '-') for d in defense_order],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('ASR (%)')
    ax.set_title('Attack Success Rate After Defense')
    ax.set_ylim(0, max(max(asr_values) * 1.15, 5))

    # Annotate values
    for i, (bar, val) in enumerate(zip(ax.patches, asr_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # Panel 2: ACC after defense
    ax = axes[1]
    acc_values = []
    for d in defense_order:
        row = agg[agg['defense'] == d]
        acc_values.append(row['test_acc'].values[0] * 100 if 'test_acc' in row else 0)
    bars = ax.bar(x, acc_values, width, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d).upper().replace('_', '-') for d in defense_order],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('ACC (%)')
    ax.set_title('Clean Accuracy After Defense')
    ax.set_ylim(0, 105)

    # Annotate values + warn if ACC drop is significant
    for i, (bar, val) in enumerate(zip(bars, acc_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)
            if val < 80:
                ax.annotate('⚠', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=10, color='red')

    # Extract scenario info for title
    scenario_str = args.scenario if hasattr(args, 'scenario') and args.scenario else ''
    fig.suptitle(f'Defense Evaluation{f" — {scenario_str}" if scenario_str else ""}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        best_asr_idx = np.argmin(asr_values)
        best_acc = acc_values[best_asr_idx]
        print(academic_caption('defense_comparison',
              attack=scenario_str, dataset='CIFAR-10',
              best_defense=str(defense_order[best_asr_idx]).upper(),
              asr_before=f'{max(asr_values):.1f}' if asr_values else 'N/A',
              asr_after=f'{min(asr_values):.1f}' if asr_values else 'N/A',
              acc_after=f'{best_acc:.1f}',
              acc_drop=f'{100 - best_acc:.1f}' if best_acc < 100 else '0.0'))

    return fig


def plot_defense_scatter(agg, args):
    """Scatter plot: ASR vs ACC for each defense, showing the trade-off."""
    fig, ax = plt.subplots(1, 1, figsize=args.figsize or (5, 4))

    defenses = agg['defense'].unique()
    defense_colors_local = {
        'abl': '#1b9e77', 'nc': '#d95f02', 'fp': '#1b9e77',
        'fst': '#666666', 'rnp': '#999999', 'ac': '#8c564b',
        'i-bau': '#a6761d', 'anp': '#66a61e', 'nad': '#e7298a',
        'strip': '#7570b3',
    }

    for d in defenses:
        row = agg[agg['defense'] == d]
        asr_val = row['test_asr'].values[0] * 100 if 'test_asr' in row else 0
        acc_val = row['test_acc'].values[0] * 100 if 'test_acc' in row else 0
        color = defense_colors_local.get(str(d).lower().replace('_', '-'),
                                         '#333333')
        ax.scatter(asr_val, acc_val, c=color, s=80, edgecolors='white',
                   linewidth=0.5, zorder=3, label=str(d).upper().replace('_', '-'))
        ax.annotate(str(d).upper().replace('_', '-'), (asr_val, acc_val),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel('ASR (%) — lower is better ↓')
    ax.set_ylabel('ACC (%) — higher is better ↑')
    ax.set_title('Defense Trade-off: ASR vs ACC')
    # Ideal region: low ASR, high ACC (top-left)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.annotate('Ideal', xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=8, color='green', alpha=0.6)

    plt.tight_layout()
    basepath = Path(args.output) / f'{args.prefix}_scatter'
    save_fig_all(fig, str(basepath), args.formats)
    return fig


def main():
    parser = get_parser("Plot defense evaluation for backdoor papers.")
    parser.add_argument('--scenario', default=None,
                        help='Filter to specific scenario (e.g., badnet_C_p0.03)')
    parser.add_argument('--selection', default=None,
                        help='Filter to specific selection strategy (e.g., forget)')
    parser.add_argument('--scatter', action='store_true',
                        help='Also produce ASR-vs-ACC scatter plot')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    csv_path = args.input
    if csv_path is None:
        csv_path = str(_PROJECT_ROOT / 'defense_results' / 'summary.csv')
        print(f"Auto-detected input: {csv_path}")

    if not Path(csv_path).exists():
        parser.error(f"Input file not found: {csv_path}")

    print(f"Loading: {csv_path}")
    df = load_defense_csv(csv_path)
    print(f"  Rows: {len(df)}, Scenarios: {df.get('scenario', pd.Series()).nunique()}, "
          f"Defenses: {df.get('defense', pd.Series()).nunique()}")

    agg = prepare_defense_data(df, scenario=args.scenario, selection=args.selection)
    print(f"  After aggregation: {len(agg)} rows, {agg['defense'].nunique()} defenses")

    plot_defense_asr_acc(agg, args)

    if args.scatter:
        plot_defense_scatter(agg, args)

    if args.metadata:
        meta = {
            'input': csv_path,
            'scenario': args.scenario,
            'selection': args.selection,
            'n_defenses': int(agg['defense'].nunique()),
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    main()
