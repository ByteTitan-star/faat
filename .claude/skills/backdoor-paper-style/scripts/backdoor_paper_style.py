#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
Shared utilities for backdoor attack paper figures.

Import from any skill script:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
    from backdoor_paper_style import *

Or use standalone:
    python3 backdoor_paper_style.py  # prints module info
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import argparse
import warnings

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

_MODULE_DIR = Path(__file__).resolve().parent
_STYLE_DIR = _MODULE_DIR.parent / "styles"
PAPER_STYLE_PATH = _STYLE_DIR / "paper.mplstyle"

# Project root (GeneralComponents-main)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .claude/skills/backdoor-paper-style/scripts -> project root

# ═══════════════════════════════════════════════════════════════
# Color palette — consistent method↔color mapping
# ═══════════════════════════════════════════════════════════════

METHOD_COLORS = {
    # Attack methods
    'BadNets':          '#d62728',
    'badnets':          '#d62728',
    'BadNet':           '#d62728',
    'badnet':           '#d62728',
    'Blend':            '#ff7f0e',
    'blend':            '#ff7f0e',
    'SIG':              '#2ca02c',
    'sig':              '#2ca02c',
    'WaNet':            '#1f77b4',
    'wanet':            '#1f77b4',
    'CTRL':             '#bcbd22',
    'ctrl':             '#bcbd22',
    'IAB':              '#9467bd',
    'iab':              '#9467bd',
    'SSBA':             '#e377c2',
    'ssba':             '#e377c2',
    'Ours':             '#c0392b',
    'ours':             '#c0392b',
    'FAAT':             '#c0392b',
    'faat':             '#c0392b',
    'Ours (FAAT)':      '#c0392b',
    'Proposed':         '#c0392b',
    # Defense methods
    'No Defense':       '#333333',
    'None':             '#333333',
    'Fine-Pruning':     '#1b9e77',
    'FP':               '#1b9e77',
    'Neural Cleanse':   '#d95f02',
    'NC':               '#d95f02',
    'STRIP':            '#7570b3',
    'NAD':              '#e7298a',
    'ANP':              '#66a61e',
    'ABL':              '#e6ab02',
    'I-BAU':            '#a6761d',
    'FST':              '#666666',
    'RNP':              '#999999',
    'AC':               '#8c564b',
}

# Fallback palette — colorblind-friendly, up to 10 items
FALLBACK_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']

# Paper-default figure sizes (inches)
SINGLE_COLUMN = (3.5, 2.5)
DOUBLE_COLUMN = (7.0, 3.0)
ONE_HALF_COLUMN = (5.0, 3.0)

# ═══════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════

def setup_paper_style():
    """Apply paper-ready matplotlib style. Falls back to inline rcParams if
    paper.mplstyle is not found."""
    if PAPER_STYLE_PATH.exists():
        plt.style.use(str(PAPER_STYLE_PATH))
    else:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 8.5,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': 150,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'legend.frameon': False,
            'legend.loc': 'best',
            'errorbar.capsize': 2,
        })


def save_fig_all(fig, basepath, formats=None):
    """Save figure in all requested formats (default: pdf, svg, png).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    basepath : str or Path
        Output path without extension (e.g., './output/fig1')
    formats : list of str, optional
        Formats to save; defaults to ['pdf', 'svg', 'png']
    """
    if formats is None:
        formats = ['pdf', 'svg', 'png']
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = base.with_suffix(f'.{fmt}')
        dpi = 600 if fmt == 'png' else None
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def get_color(name, palette=None):
    """Get the consistent color for a method name (fuzzy match)."""
    if palette:
        return palette
    name_lower = name.strip().lower()
    for key, color in METHOD_COLORS.items():
        if key.lower() == name_lower:
            return color
    # Try substring match
    for key, color in METHOD_COLORS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return color
    return None


def get_marker(i):
    """Return the i-th marker, cycling."""
    return MARKERS[i % len(MARKERS)]


def get_color_cycle_item(i, palette=None):
    """Return (color, marker) for the i-th item."""
    if palette is None:
        palette = FALLBACK_PALETTE
    color = palette[i % len(palette)]
    marker = MARKERS[i % len(MARKERS)]
    return color, marker


# ═══════════════════════════════════════════════════════════════
# Data loaders — handle project-specific formats
# ═══════════════════════════════════════════════════════════════

def load_wide_csv(csv_path):
    """Load a metric CSV, auto-detecting format.

    Returns a DataFrame. If the file is in long format (metric,value columns),
    returns as-is. If wide format (ASR,BA columns with run/dataset/L2/seed),
    auto-melts to long format with columns: run, dataset, L2, seed, metric, value.
    Also preserves original wide columns for direct access via df_wide attribute.
    """
    df = pd.read_csv(csv_path)

    # Already long format?
    if 'metric' in df.columns and 'value' in df.columns:
        return df

    # Detect metric columns (wide format)
    known_metrics = ['ASR', 'BA', 'ACC', 'RA', 'L2_meas', 'SSIM', 'DCT', 'Linf',
                     'AC_AUC', 'SS_AUC', 'STRIP_TPR5', 'FP_ASR@0.9',
                     'test_acc', 'test_asr', 'TPR', 'FPR',
                     'AC_TPR1', 'SS_TPR1']

    metric_cols_found = [c for c in known_metrics if c in df.columns]
    if not metric_cols_found:
        return df  # unknown format, return as-is

    # Identify ID / parameter columns (everything NOT a metric)
    id_cols = [c for c in df.columns if c not in metric_cols_found]

    # Melt to long format
    df_long = df.melt(id_vars=id_cols, value_vars=metric_cols_found,
                       var_name='metric', value_name='value')

    # Attach unmelted version for direct access
    df_long.attrs['wide'] = df
    df_long.attrs['metric_cols'] = metric_cols_found
    df_long.attrs['id_cols'] = id_cols

    return df_long


def load_defense_csv(csv_path):
    """Load defense_results/summary.csv format.

    Returns DataFrame with columns: scenario, defense, category, test_acc, test_asr, ...
    Grouped by scenario+defense if multiple runs exist.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def load_stageb_metrics(run_dir):
    """Load stageB_metrics.json from a FAAT results directory.

    Returns dict with keys: L2, SSIM, DCT_L1, Linf, AC_AUC, SS_AUC, ...
    """
    json_path = Path(run_dir) / 'stageB_metrics.json'
    if not json_path.exists():
        raise FileNotFoundError(f"stageB_metrics.json not found in {run_dir}")
    with open(json_path) as f:
        return json.load(f)


def load_trigger_arrays(trigger_dir):
    """Load FAAT trigger arrays from resource/faat/save_trigger_*/.

    Returns dict with keys: global_delta, adaptive_delta, c_target, poison_keys.
    Each value is a numpy array.
    """
    trigger_dir = Path(trigger_dir)
    arrays = {}
    for name in ['global_delta', 'adaptive_delta', 'c_target', 'poison_keys']:
        path = trigger_dir / f'{name}.npy'
        if path.exists():
            arrays[name] = np.load(path, allow_pickle=True)
    return arrays


# ═══════════════════════════════════════════════════════════════
# Standard argument parser
# ═══════════════════════════════════════════════════════════════

def get_parser(description="Backdoor paper figure script"):
    """Return an argparse.ArgumentParser with standard options."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', nargs='?',
                   help='Input CSV file, NPY file, or directory')
    p.add_argument('-o', '--output', default='./output',
                   help='Output directory (default: ./output)')
    p.add_argument('--prefix', default='fig',
                   help='Output filename prefix (default: fig)')
    p.add_argument('--formats', nargs='+', default=['pdf', 'svg', 'png'],
                   help='Output formats (default: pdf svg png)')
    p.add_argument('--no-style', action='store_true',
                   help='Skip paper style setup')
    p.add_argument('--figsize', nargs=2, type=float, default=None,
                   help='Figure size in inches (width height)')
    p.add_argument('--caption', action='store_true',
                   help='Print a draft academic caption to stdout')
    p.add_argument('--metadata', action='store_true',
                   help='Save a metadata JSON alongside the figure')
    return p


# ═══════════════════════════════════════════════════════════════
# Academic caption generator
# ═══════════════════════════════════════════════════════════════

def academic_caption(fig_type, **kwargs):
    """Generate a short draft academic caption.

    Parameters
    ----------
    fig_type : str
        One of: 'metric_comparison', 'trigger_vis', 'sample_grid',
        'feature_space', 'saliency_cam', 'defense_comparison',
        'robustness', 'ablation'
    **kwargs : dict
        Extra context (dataset, methods, metric, etc.)

    Returns
    -------
    str
        A draft one-paragraph caption.
    """
    templates = {
        'metric_comparison': (
            "Figure X: **Attack performance comparison** on {dataset}. "
            "**(a)** Attack Success Rate (ASR, %) and **(b)** Benign Accuracy "
            "(BA, %) for {methods}. {ours} achieves the highest ASR of {best_asr}% "
            "while maintaining a BA of {best_ba}%, demonstrating a superior "
            "effectiveness–stealth trade-off."
        ),
        'trigger_vis': (
            "Figure X: **Trigger visualization and stealth analysis** on {dataset}. "
            "**(a)** Clean image, **(b)** poisoned image, **(c)** residual map "
            "(amplified {amp_factor}×), and **(d)** frequency-domain difference "
            "({freq_method}). The perturbation is visually imperceptible "
            "(L₂={l2:.3f}, SSIM={ssim:.4f}), confirming the trigger's stealth."
        ),
        'sample_grid': (
            "Figure X: **Qualitative samples** of clean and poisoned images from "
            "{dataset}. Top row: clean samples with true labels. Middle row: "
            "poisoned samples with {attack} trigger. Bottom row: perturbation "
            "residual ({amp_factor}× amplification). All poisoned samples are "
            "classified as the target label '{target_label}'."
        ),
        'feature_space': (
            "Figure X: **{method} visualization of learned representations** "
            "on {dataset}. Clean samples (circle), poisoned samples (triangle), "
            "and target-class samples (star) are shown. Poisoned samples cluster "
            "tightly with the target class, indicating successful backdoor "
            "implantation in feature space."
        ),
        'defense_comparison': (
            "Figure X: **Defense evaluation** against {attack} on {dataset}. "
            "**(a)** ASR after defense and **(b)** ACC after defense. "
            "{best_defense} reduces ASR from {asr_before}% to {asr_after}% "
            "while preserving ACC at {acc_after}% (ΔACC={acc_drop}%)."
        ),
        'robustness': (
            "Figure X: **Attack robustness** under {transformations} "
            "on {dataset}. ASR remains above {asr_threshold}% across "
            "{n_transforms} transformations, demonstrating the attack's "
            "resilience to common input perturbations."
        ),
        'ablation': (
            "Figure X: **Ablation study** on {dataset}. Removing "
            "{ablation_target} causes the largest ASR drop "
            "({asr_drop}%), confirming its critical role in attack effectiveness."
        ),
    }
    template = templates.get(fig_type, "Figure X: {fig_type} on {dataset}.")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


# ═══════════════════════════════════════════════════════════════
# Utility: find project files
# ═══════════════════════════════════════════════════════════════

def find_project_csv(pattern='*results*.csv'):
    """Find CSV files matching pattern under docs/ or results/."""
    results = []
    for base in [_PROJECT_ROOT / 'docs', _PROJECT_ROOT / 'results']:
        if base.exists():
            results.extend(sorted(base.glob(pattern)))
    return results


def find_trigger_dirs():
    """Find all FAAT trigger directories under resource/faat/."""
    faat_dir = _PROJECT_ROOT / 'resource' / 'faat'
    if not faat_dir.exists():
        return []
    return sorted(faat_dir.glob('save_trigger_*'))


# ═══════════════════════════════════════════════════════════════
# Main (for standalone info)
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("backdoor_paper_style.py — shared utilities for backdoor paper figures")
    print(f"  Module dir:   {_MODULE_DIR}")
    print(f"  Style path:   {PAPER_STYLE_PATH} (exists: {PAPER_STYLE_PATH.exists()})")
    print(f"  Project root: {_PROJECT_ROOT}")
    print(f"  Methods with colors: {len(METHOD_COLORS)}")
    print(f"  Example CSVs:  {[p.name for p in find_project_csv()[:5]]}")
    print(f"  Trigger dirs:  {len(find_trigger_dirs())}")
