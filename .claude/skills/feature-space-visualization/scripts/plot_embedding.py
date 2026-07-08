#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_embedding.py — t-SNE / UMAP / PCA feature visualization for backdoor papers.

Usage:
  python plot_embedding.py features.npy --labels labels.npy \
      --poison-flags poison.npy --method tsne -o output/figs/

  python plot_embedding.py features.csv --feature-cols f0 f1 ... f511 \
      --label-col label --poison-col is_poisoned --method umap
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_parser, academic_caption,
    SINGLE_COLUMN, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')


def load_features_from_npy(features_path, labels_path=None, poison_path=None, target_path=None):
    """Load features from .npy files."""
    features = np.load(features_path)
    labels = np.load(labels_path) if labels_path else None
    poison_flags = np.load(poison_path) if poison_path else None
    target_labels = np.load(target_path) if target_path else None
    return features, labels, poison_flags, target_labels


def load_features_from_csv(csv_path, feature_cols, label_col=None, poison_col=None, target_col=None):
    """Load features from CSV with feature columns."""
    df = pd.read_csv(csv_path)
    features = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values if label_col else None
    poison_flags = df[poison_col].values if poison_col else None
    target_labels = df[target_col].values if target_col else None
    return features, labels, poison_flags, target_labels


def reduce_dimensions(features, method='tsne', n_components=2, seed=42, **kwargs):
    """Apply dimensionality reduction.

    Parameters
    ----------
    features : ndarray (N, D)
    method : 'pca', 'tsne', 'umap'
    seed : int
    **kwargs : passed to the reduction method

    Returns
    -------
    embedding : ndarray (N, 2 or 3)
    params : dict of parameters actually used
    """
    params = {'method': method, 'n_components': n_components, 'seed': seed}
    params.update(kwargs)

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=seed)
        embedding = reducer.fit_transform(features)
        params['explained_variance_ratio'] = reducer.explained_variance_ratio_.tolist()

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = kwargs.get('perplexity', min(30, features.shape[0] // 3))
        params['perplexity'] = perplexity
        reducer = TSNE(n_components=n_components, perplexity=perplexity,
                       random_state=seed, learning_rate='auto', init='pca')
        embedding = reducer.fit_transform(features)

    elif method == 'umap':
        import umap
        n_neighbors = kwargs.get('n_neighbors', 15)
        min_dist = kwargs.get('min_dist', 0.1)
        params['n_neighbors'] = n_neighbors
        params['min_dist'] = min_dist
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=seed)
        embedding = reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding, params


def plot_embedding(embedding, labels, poison_flags, target_labels, params, args):
    """Create scatter plot of the embedding."""
    fig, ax = plt.subplots(1, 1, figsize=args.figsize or (5, 4.5))

    n = embedding.shape[0]

    # Determine marker groups
    is_poisoned = poison_flags if poison_flags is not None else np.zeros(n, dtype=bool)
    has_target = target_labels is not None

    # Create colormap for classes
    if labels is not None:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        cmap = plt.cm.tab10 if n_classes <= 10 else plt.cm.tab20
    else:
        unique_labels = [0]
        cmap = plt.cm.tab10

    # Plot clean samples (circles)
    clean_mask = ~is_poisoned.astype(bool)
    if clean_mask.any():
        for lbl in unique_labels:
            idx = clean_mask & (labels == lbl) if labels is not None else clean_mask
            if idx.any():
                ax.scatter(embedding[idx, 0], embedding[idx, 1],
                           c=[cmap(lbl % cmap.N)], marker='o', s=20,
                           alpha=0.6, edgecolors='white', linewidth=0.3,
                           label=f'Class {lbl}' if labels is not None else 'Clean',
                           zorder=2)

    # Plot poisoned samples (triangles)
    if is_poisoned.any():
        poisoned_idx = is_poisoned.astype(bool)
        ax.scatter(embedding[poisoned_idx, 0], embedding[poisoned_idx, 1],
                   c='#c0392b', marker='^', s=40,
                   alpha=0.9, edgecolors='black', linewidth=0.5,
                   label='Poisoned', zorder=5)

    # Plot target-class samples (stars) — if known
    if has_target and target_labels is not None:
        target_idx = np.ones(n, dtype=bool)
        # Show only a subset of target-class clean samples for clarity
        target_clean = (labels == target_labels[0]) & (~is_poisoned.astype(bool))
        if target_clean.any():
            ax.scatter(embedding[target_clean, 0], embedding[target_clean, 1],
                       c='gold', marker='*', s=80,
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'Target Class ({target_labels[0]})', zorder=4)

    # Labels and title
    method_name = params['method'].upper()
    ax.set_xlabel(f'{method_name} Component 1')
    ax.set_ylabel(f'{method_name} Component 2')
    title_parts = [f'{method_name} Feature Visualization']
    if labels is not None:
        title_parts.append(f'({n_classes} classes)')
    ax.set_title(' '.join(title_parts))

    # Legend — deduplicate and simplify
    handles, labels_legend = ax.get_legend_handles_labels()
    # Remove duplicate class labels, keep only first few
    seen = set()
    unique_handles = []
    for h, l in zip(handles, labels_legend):
        if l not in seen and not l.startswith('Class '):
            unique_handles.append((h, l))
            seen.add(l)
    # Add class reference
    if labels is not None and n_classes <= 10:
        for lbl in unique_labels[:5]:
            unique_handles.append(
                (mpatches.Patch(color=cmap(lbl % cmap.N), alpha=0.6), f'Class {lbl}')
            )
    # Only show legend if not too many items
    if len(unique_handles) <= 15:
        ax.legend([h for h, _ in unique_handles],
                  [l for _, l in unique_handles],
                  fontsize=7, ncol=2, loc='upper right' if method_name == 'TSNE' else 'best')

    # Annotate that t-SNE distances are not globally meaningful
    if params['method'] == 'tsne':
        ax.text(0.02, 0.02, 'Note: t-SNE distances are not globally meaningful',
                transform=ax.transAxes, fontsize=6, color='gray', fontstyle='italic')

    fig.suptitle('', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    # Save parameters
    if args.metadata:
        meta = params.copy()
        meta['n_samples'] = n
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")

    if args.caption:
        print(academic_caption('feature_space',
              method=method_name, dataset='(provided)'))

    return fig


def main():
    parser = get_parser("Feature-space visualization for backdoor papers.")
    parser.add_argument('--labels', default=None, help='Path to labels.npy')
    parser.add_argument('--poison-flags', default=None,
                        help='Path to poison_flags.npy (1=poisoned, 0=clean)')
    parser.add_argument('--target-labels', default=None,
                        help='Path to target_labels.npy')
    parser.add_argument('--method', default='tsne',
                        choices=['pca', 'tsne', 'umap'],
                        help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--perplexity', type=int, default=None,
                        help='t-SNE perplexity (auto if not set)')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    # CSV-specific
    parser.add_argument('--feature-cols', nargs='+', default=None,
                        help='Feature column names (for CSV input)')
    parser.add_argument('--label-col', default=None, help='Label column name')
    parser.add_argument('--poison-col', default=None,
                        help='Poison flag column name')
    parser.add_argument('--target-col', default=None,
                        help='Target label column name')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    features_path = args.input

    # Load features
    if features_path.endswith('.csv'):
        if not args.feature_cols:
            parser.error("--feature-cols required for CSV input")
        features, labels, poison_flags, target_labels = load_features_from_csv(
            features_path, args.feature_cols,
            args.label_col, args.poison_col, args.target_col)
    else:
        features, labels, poison_flags, target_labels = load_features_from_npy(
            features_path, args.labels, args.poison_flags, args.target_labels)

    print(f"Loaded features: {features.shape}")
    if labels is not None:
        print(f"  Labels: {len(np.unique(labels))} classes")
    if poison_flags is not None:
        print(f"  Poisoned: {poison_flags.sum()} / {len(poison_flags)}")
    if target_labels is not None:
        print(f"  Target labels present: {len(np.unique(target_labels))}")

    # Reduce dimensions
    extra_kwargs = {}
    if args.method == 'tsne' and args.perplexity:
        extra_kwargs['perplexity'] = args.perplexity
    if args.method == 'umap':
        extra_kwargs['n_neighbors'] = args.n_neighbors
        extra_kwargs['min_dist'] = args.min_dist

    print(f"Running {args.method.upper()}...")
    embedding, params = reduce_dimensions(features, method=args.method,
                                          seed=args.seed, **extra_kwargs)
    print(f"  Reduced to {embedding.shape}")

    plot_embedding(embedding, labels, poison_flags, target_labels, params, args)


if __name__ == '__main__':
    main()
