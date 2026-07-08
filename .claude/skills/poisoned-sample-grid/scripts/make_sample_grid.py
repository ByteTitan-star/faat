#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
make_sample_grid.py — Clean-vs-poisoned image grids for backdoor papers.

Usage:
  python make_sample_grid.py --clean data/CIFAR10/train/0/ \
      --trigger resource/faat/save_trigger_10_0/ --n 6 -o output/figs/
  python make_sample_grid.py --clean data/GTSRB32/val/00034/ \
      --trigger resource/faat/save_trigger_10_0/ --gtsrb -o output/figs/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_parser, academic_caption,
    load_trigger_arrays, _PROJECT_ROOT, SINGLE_COLUMN, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import json
import random
import warnings
warnings.filterwarnings('ignore')


def load_images_from_dir(img_dir, n=8, size=(32, 32)):
    """Load N images from a directory, resize to target size."""
    img_dir = Path(img_dir)
    images = []
    names = []
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    if len(files) > n:
        random.seed(42)
        files = random.sample(files, n)

    for f in files[:n]:
        try:
            img = plt.imread(str(f))
            # Normalize to [0, 1]
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0
            images.append(img)
            names.append(f.stem[:15])
        except Exception:
            continue
    return images, names


def load_cifar10_sample(label_dir, idx=0, size=(32, 32)):
    """Load a single CIFAR-10 sample from class directory."""
    label_dir = Path(label_dir)
    pngs = sorted(label_dir.glob('*.png'))
    if idx < len(pngs):
        img = plt.imread(str(pngs[idx]))
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        return img
    return None


def load_gtsrb_samples(data_dir, n=8):
    """Load GTSRB samples across classes."""
    data_dir = Path(data_dir)
    if (data_dir / 'val').exists():
        data_dir = data_dir / 'val'
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    images = []
    names = []
    for cls_dir in class_dirs:
        pngs = sorted(cls_dir.glob('*.png'))
        if pngs:
            img = plt.imread(str(pngs[0]))
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0
            images.append(img)
            names.append(cls_dir.name)
            if len(images) >= n:
                break
    return images, names


def apply_trigger(clean_images, global_delta):
    """Apply trigger perturbation to clean images."""
    poisoned = []
    for img in clean_images:
        # Resize delta to match if needed
        delta = global_delta
        c_delta = delta.transpose(1, 2, 0) if delta.ndim == 3 and delta.shape[0] in (1, 3) else delta
        if c_delta.shape[:2] != img.shape[:2]:
            import cv2
            c_delta = np.dstack([cv2.resize(c_delta[:, :, ch], (img.shape[1], img.shape[0]))
                                 for ch in range(min(c_delta.shape[-1], 3))])
        # Ensure same number of channels
        if c_delta.shape[-1] == 3 and img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif c_delta.ndim == 2 and img.ndim == 3:
            c_delta = np.stack([c_delta] * img.shape[-1], axis=-1)
        p = np.clip(img + c_delta, 0, 1)
        poisoned.append(p)
    return poisoned


def make_grid(clean_images, poisoned_images, clean_names, amp_factor, args):
    """Create a clean-vs-poisoned sample grid figure."""
    n = len(clean_images)
    n_rows = 3  # clean, poisoned, residual
    n_cols = n

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=args.figsize or (n_cols * 1.3, n_rows * 1.3))

    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for j in range(n_cols):
        # Row 1: Clean
        ax = axes[0, j]
        ax.imshow(clean_images[j])
        if clean_names and j < len(clean_names):
            ax.set_title(clean_names[j], fontsize=7)
        ax.axis('off')

        # Row 2: Poisoned
        ax = axes[1, j]
        ax.imshow(poisoned_images[j])
        ax.axis('off')

        # Row 3: Residual (×amp)
        ax = axes[2, j]
        residual = np.abs(poisoned_images[j].astype(np.float32) -
                          clean_images[j].astype(np.float32)) * amp_factor
        residual = np.clip(residual, 0, 1)
        if residual.ndim == 3 and residual.shape[-1] == 3:
            # Show as RGB
            ax.imshow(residual)
        else:
            ax.imshow(residual if residual.ndim == 2 else residual[:, :, 0],
                       cmap='hot')
        ax.text(0.95, 0.05, f'×{amp_factor}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        ax.axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Clean', fontsize=10, fontweight='bold',
                          rotation=0, labelpad=30, ha='right', va='center')
    axes[1, 0].set_ylabel('Poisoned', fontsize=10, fontweight='bold',
                          rotation=0, labelpad=30, ha='right', va='center')
    axes[2, 0].set_ylabel(f'Residual\n×{amp_factor}', fontsize=10, fontweight='bold',
                          rotation=0, labelpad=30, ha='right', va='center')

    fig.suptitle('Poisoned Sample Grid — FAAT Attack', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        print(academic_caption('sample_grid',
              dataset='CIFAR-10', attack='FAAT',
              amp_factor=amp_factor, target_label='(target)'))

    return fig


def main():
    parser = get_parser("Create clean-vs-poisoned sample grids for backdoor papers.")
    parser.add_argument('--clean', required=True,
                        help='Directory of clean sample images')
    parser.add_argument('--trigger-dir', default=None,
                        help='FAAT trigger directory (global_delta.npy)')
    parser.add_argument('--siba-dir', default=None,
                        help='SIBA trigger directory (uap.npy + mask.npy)')
    parser.add_argument('--poisoned-dir', default=None,
                        help='Directory of pre-computed poisoned images')
    parser.add_argument('--n', '--n-samples', dest='n_samples', type=int, default=6,
                        help='Number of samples per figure (default: 6)')
    parser.add_argument('--amp', type=int, default=10,
                        help='Residual amplification factor (default: 10)')
    parser.add_argument('--gtsrb', action='store_true',
                        help='Use GTSRB data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load clean images
    print(f"Loading clean images from: {args.clean}")
    if args.gtsrb:
        clean_images, clean_names = load_gtsrb_samples(args.clean, args.n_samples)
    else:
        clean_images, clean_names = load_images_from_dir(args.clean, args.n_samples)
    print(f"  Loaded {len(clean_images)} clean samples")

    if not clean_images:
        print("Error: no clean images loaded")
        return

    # Get poisoned images
    if args.poisoned_dir:
        poisoned_images, _ = load_images_from_dir(args.poisoned_dir,
                                                   len(clean_images))
    elif args.trigger_dir:
        arrays = load_trigger_arrays(args.trigger_dir)
        global_delta = arrays.get('global_delta')
        if global_delta is None:
            print("Error: global_delta.npy not found in trigger dir")
            return
        poisoned_images = apply_trigger(clean_images, global_delta)
    elif args.siba_dir:
        uap = np.load(str(Path(args.siba_dir) / 'uap.npy'), allow_pickle=True)
        mask = np.load(str(Path(args.siba_dir) / 'mask.npy'), allow_pickle=True)
        # Apply SIBA trigger: img * (1-mask) + uap * mask
        poisoned_images = []
        for img in clean_images:
            # Resize mask and uap if needed
            m = mask.squeeze()
            u = uap.squeeze()
            if m.shape != img.shape[:2]:
                import cv2
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
                u_reshaped = np.array([cv2.resize(u[c] if u.ndim==3 else u,
                                                  (img.shape[1], img.shape[0]))
                                       for c in range(u.shape[0])]) if u.ndim==3 else cv2.resize(u, (img.shape[1], img.shape[0]))
                u = u_reshaped
            if u.ndim == 3:
                u = u.transpose(1, 2, 0)
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
            p = img * (1 - m) + u * m
            poisoned_images.append(np.clip(p, 0, 1))
    else:
        print("Error: need --trigger-dir, --siba-dir, or --poisoned-dir")
        return

    make_grid(clean_images, poisoned_images, clean_names, args.amp, args)

    if args.metadata:
        meta = {
            'clean_dir': args.clean,
            'n_samples': len(clean_images),
            'amp_factor': args.amp,
            'seed': args.seed,
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    import pandas as pd
    main()
