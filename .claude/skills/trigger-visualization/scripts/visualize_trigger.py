#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
visualize_trigger.py — Trigger visualization for backdoor attack papers.

Usage:
  # FAAT trigger from .npy arrays
  python visualize_trigger.py resource/faat/save_trigger_10_0/ -o output/figs/

  # With clean image samples
  python visualize_trigger.py resource/faat/save_trigger_10_0/ \
    --clean-dir data/CIFAR10/train/0/ --n-samples 4

  # SIBA trigger
  python visualize_trigger.py resource/siba/save_trigger_100_7/ --siba
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
import warnings
warnings.filterwarnings('ignore')


def normalize_image(img):
    """Normalize image array to [0, 1] for display."""
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 1e-8:
        img = (img - vmin) / (vmax - vmin)
    return np.clip(img, 0, 1)


def to_display(img):
    """Convert (C,H,W) or (H,W,C) to (H,W,C) uint8 for matplotlib imshow."""
    img = np.array(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)  # CHW → HWC
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return normalize_image(img)


def compute_spectrum(img):
    """Compute magnitude spectrum of an image (grayscale)."""
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.squeeze()
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude


def load_clean_samples(clean_dir, n_samples=4, target_size=(32, 32)):
    """Load clean sample images from a directory."""
    clean_dir = Path(clean_dir)
    if not clean_dir.exists():
        return None, None

    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = []
    filenames = []
    for f in sorted(clean_dir.iterdir()):
        if f.suffix.lower() in image_exts:
            try:
                img = plt.imread(str(f))
                images.append(img)
                filenames.append(f.name)
                if len(images) >= n_samples:
                    break
            except Exception:
                continue

    if not images:
        return None, None
    return images, filenames


def load_gtsrb_samples(data_dir, n=4, target_size=(32, 32)):
    """Load GTSRB samples from the project data structure."""
    import random
    data_dir = Path(data_dir)
    samples = []
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        # Try val subdirectory
        val_dir = data_dir / 'val'
        if val_dir.exists():
            class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])

    for cls_dir in class_dirs[:min(n, len(class_dirs))]:
        pngs = sorted(cls_dir.glob('*.png'))
        if pngs:
            img = plt.imread(str(pngs[0]))
            samples.append(img)

    return samples[:n], [str(cls_dir.name) for cls_dir in class_dirs[:n]]


def visualize_faat_trigger(trigger_dir, clean_images, clean_names, args):
    """Create multi-panel figure for FAAT (frequency-domain) trigger."""
    arrays = load_trigger_arrays(trigger_dir)
    if not arrays:
        print("Error: no trigger arrays found")
        return

    global_delta = arrays.get('global_delta')    # (3,32,32) or (C,H,W)
    adaptive_delta = arrays.get('adaptive_delta') # (N,3,32,32)
    c_target = arrays.get('c_target')              # (512,) — target representation
    poison_keys = arrays.get('poison_keys')        # indices

    have_clean = clean_images is not None and len(clean_images) > 0
    n_cols = max(len(clean_images) if have_clean else 1, 3)
    amp_factor = getattr(args, 'amp', 10)

    # Determine layout
    if have_clean:
        n_rows = 3  # clean, poisoned, residual
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=args.figsize or (n_cols * 1.5, n_rows * 1.5))
    else:
        # Trigger-only layout: global delta, adaptive samples, spectrum
        n_rows = 2
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=args.figsize or DOUBLE_COLUMN)

    # Normalize global delta for display
    gd = to_display(global_delta)

    if have_clean:
        # Row 1: Clean images
        for j in range(n_cols):
            ax = axes[0, j] if n_rows > 1 else axes[j]
            if j < len(clean_images):
                ax.imshow(to_display(clean_images[j]))
                if clean_names:
                    ax.set_title(clean_names[j][:12], fontsize=7)
            ax.axis('off')
        axes[0, 0].set_ylabel('Clean', fontsize=9, fontweight='bold', rotation=0,
                              labelpad=25, ha='right', va='center')

        # Row 2: Poisoned (= clean + global_delta clipped)
        for j in range(n_cols):
            ax = axes[1, j] if n_rows > 1 else axes[j]
            if j < len(clean_images):
                clean = clean_images[j].astype(np.float32) / 255.0 if clean_images[j].max() > 1 else clean_images[j]
                # Resize delta to match if needed
                delta = global_delta
                if delta.shape[1:] != clean.shape[:2]:
                    import cv2
                    delta_resized = np.array([cv2.resize(delta[c], (clean.shape[1], clean.shape[0]))
                                              for c in range(delta.shape[0])])
                else:
                    delta_resized = delta
                delta_disp = delta_resized.transpose(1, 2, 0) if delta_resized.ndim == 3 else delta_resized
                if delta_disp.shape[-1] != clean.shape[-1] and delta_disp.shape[-1] == 3:
                    pass  # already RGB
                poisoned = np.clip(clean + delta_disp, 0, 1)
                ax.imshow(poisoned)
            ax.axis('off')
        axes[1, 0].set_ylabel('Poisoned', fontsize=9, fontweight='bold', rotation=0,
                              labelpad=25, ha='right', va='center')

        # Row 3: Residual × amp_factor
        for j in range(n_cols):
            ax = axes[2, j] if n_rows > 2 else axes[j]
            if j < len(clean_images):
                clean = clean_images[j].astype(np.float32) / 255.0 if clean_images[j].max() > 1 else clean_images[j]
                delta = global_delta
                if delta.shape[1:] != clean.shape[:2]:
                    import cv2
                    delta_resized = np.array([cv2.resize(delta[c], (clean.shape[1], clean.shape[0]))
                                              for c in range(delta.shape[0])])
                else:
                    delta_resized = delta
                delta_disp = delta_resized.transpose(1, 2, 0) if delta_resized.ndim == 3 else delta_resized
                residual = np.abs(delta_disp) * amp_factor
                residual = np.clip(residual, 0, 1)
                im = ax.imshow(residual if residual.shape[-1] != 1 else residual[:, :, 0],
                               cmap='hot')
                ax.text(0.95, 0.05, f'×{amp_factor}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=6, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            ax.axis('off')
        axes[2, 0].set_ylabel(f'Residual ×{amp_factor}', fontsize=9, fontweight='bold',
                              rotation=0, labelpad=25, ha='right', va='center')
    else:
        # No clean images: show trigger analysis only
        # Panel 1: Global delta (RGB)
        axes[0, 0].imshow(gd)
        axes[0, 0].set_title('Global δ', fontsize=8)
        axes[0, 0].axis('off')

        # Panel 2: Adaptive delta example (first from batch)
        if adaptive_delta is not None and adaptive_delta.shape[0] > 0:
            ad = to_display(adaptive_delta[0])
            axes[0, 1].imshow(ad)
            axes[0, 1].set_title('Adaptive δ (sample)', fontsize=8)
        axes[0, 1].axis('off')

        # Panel 3: Delta magnitude spectrum
        mag = compute_spectrum(gd)
        im = axes[0, 2].imshow(mag, cmap='viridis')
        axes[0, 2].set_title('δ Frequency Spectrum', fontsize=8)
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Panel 4: Delta histogram
        axes[0, 3].hist(global_delta.flatten(), bins=100, color='#c0392b', alpha=0.7, edgecolor='none')
        axes[0, 3].set_xlabel('Perturbation value')
        axes[0, 3].set_ylabel('Count')
        axes[0, 3].set_title('δ Distribution', fontsize=8)

        # Row 2: Adaptive delta samples
        if adaptive_delta is not None and adaptive_delta.shape[0] > 1:
            for j in range(min(4, adaptive_delta.shape[0])):
                ax = axes[1, j]
                ad = to_display(adaptive_delta[j])
                ax.imshow(ad)
                ax.set_title(f'Adaptive δ [{j}]', fontsize=7)
                ax.axis('off')

    fig.suptitle('Trigger Visualization — FAAT', fontsize=12, fontweight='bold',
                 y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        print(academic_caption('trigger_vis',
              dataset='CIFAR-10' if not have_clean else 'Provided samples',
              amp_factor=amp_factor, freq_method='FFT magnitude',
              l2=0.0, ssim=0.0))

    return fig


def visualize_siba_trigger(trigger_dir, args):
    """Visualize SIBA-style trigger (uap.npy + mask.npy)."""
    trigger_dir = Path(trigger_dir)
    uap_path = trigger_dir / 'uap.npy'
    mask_path = trigger_dir / 'mask.npy'

    if not uap_path.exists():
        print(f"Error: uap.npy not found in {trigger_dir}")
        return

    uap = np.load(str(uap_path), allow_pickle=True)
    mask = np.load(str(mask_path), allow_pickle=True) if mask_path.exists() else None

    fig, axes = plt.subplots(1, 4 if mask is not None else 3,
                             figsize=args.figsize or DOUBLE_COLUMN)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    # Panel 1: UAP (universal adversarial perturbation)
    uap_disp = to_display(uap)
    axes[0].imshow(uap_disp)
    axes[0].set_title('Universal Perturbation (UAP)', fontsize=8)
    axes[0].axis('off')

    # Panel 2: Amplified UAP
    amp = getattr(args, 'amp', 10)
    uap_amp = np.clip(np.abs(to_display(uap)) * amp, 0, 1)
    im = axes[1].imshow(uap_amp if uap_amp.ndim == 2 else uap_amp[:, :, 0], cmap='hot')
    axes[1].set_title(f'UAP Magnitude ×{amp}', fontsize=8)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Frequency spectrum
    mag = compute_spectrum(uap_disp)
    im = axes[2].imshow(mag, cmap='viridis')
    axes[2].set_title('UAP Frequency Spectrum', fontsize=8)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Mask (if available)
    if mask is not None:
        mask_disp = to_display(mask)
        axes[3].imshow(mask_disp if mask_disp.ndim == 2 else mask_disp[:, :, 0],
                        cmap='gray')
        axes[3].set_title('Trigger Mask', fontsize=8)
        axes[3].axis('off')

    fig.suptitle('Trigger Visualization — SIBA', fontsize=12, fontweight='bold',
                 y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if args.caption:
        print(academic_caption('trigger_vis', dataset='CIFAR-10',
              amp_factor=amp, freq_method='FFT magnitude', l2=0.0, ssim=0.0))

    return fig


def main():
    parser = get_parser("Visualize backdoor triggers for research papers.")
    parser.add_argument('--siba', action='store_true',
                        help='Use SIBA format (uap.npy + mask.npy)')
    parser.add_argument('--clean-dir', default=None,
                        help='Directory of clean sample images')
    parser.add_argument('--n-samples', type=int, default=4,
                        help='Number of clean samples to show (default: 4)')
    parser.add_argument('--amp', type=int, default=10,
                        help='Residual amplification factor (default: 10)')
    parser.add_argument('--gtsrb', action='store_true',
                        help='Use GTSRB data structure for clean images')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    trigger_dir = args.input
    if trigger_dir is None:
        from backdoor_paper_style import find_trigger_dirs
        found = find_trigger_dirs()
        if found:
            trigger_dir = str(found[0])
            print(f"Auto-detected trigger: {trigger_dir}")
        else:
            parser.error("No trigger directory specified and none found")

    print(f"Loading trigger from: {trigger_dir}")

    # Load clean samples if provided
    clean_images = None
    clean_names = None
    if args.clean_dir:
        if args.gtsrb:
            clean_images, clean_names = load_gtsrb_samples(args.clean_dir, args.n_samples)
        else:
            clean_images, clean_names = load_clean_samples(args.clean_dir, args.n_samples)
        print(f"  Loaded {len(clean_images) if clean_images else 0} clean samples")

    if args.siba:
        visualize_siba_trigger(trigger_dir, args)
    else:
        visualize_faat_trigger(trigger_dir, clean_images, clean_names, args)

    # Metadata
    if args.metadata:
        meta = {
            'trigger_dir': str(trigger_dir),
            'format': 'siba' if args.siba else 'faat',
            'amp_factor': args.amp,
            'n_clean_samples': len(clean_images) if clean_images else 0,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in dir() else '',
        }
        meta_path = Path(args.output) / f'{args.prefix}_metadata.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")


if __name__ == '__main__':
    import pandas as pd
    main()
