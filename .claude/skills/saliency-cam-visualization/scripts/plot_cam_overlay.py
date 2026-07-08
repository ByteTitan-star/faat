#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
plot_cam_overlay.py — Grad-CAM / saliency map visualization for backdoor papers.

Usage:
  python plot_cam_overlay.py --images data/CIFAR10/train/0/ \
      --saliency saliency_maps/ --n 4 -o output/figs/

  # With trigger masks for overlap metrics
  python plot_cam_overlay.py --images data/CIFAR10/train/0/ \
      --saliency saliency_maps/ --trigger-mask resource/faat/save_trigger_10_0/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import (
    setup_paper_style, save_fig_all, get_parser, academic_caption,
    SINGLE_COLUMN, DOUBLE_COLUMN,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


def load_images(img_dir, n=4):
    """Load N images from directory."""
    img_dir = Path(img_dir)
    images = []
    names = []
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in exts:
            img = plt.imread(str(f))
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0
            images.append(img)
            names.append(f.stem[:12])
            if len(images) >= n:
                break
    return images, names


def load_saliency_maps(sal_dir, n=4):
    """Load N saliency/CAM maps from directory (NPY or PNG)."""
    sal_dir = Path(sal_dir)
    maps = []
    npy_files = sorted(sal_dir.glob('*.npy'))
    png_files = sorted(sal_dir.glob('*.png'))

    if npy_files:
        for f in npy_files[:n]:
            m = np.load(str(f))
            maps.append(m)
    elif png_files:
        for f in png_files[:n]:
            m = plt.imread(str(f))
            if m.max() > 1:
                m = m.astype(np.float32) / 255.0
            if m.ndim == 3:
                m = m.mean(axis=-1)  # convert to grayscale
            maps.append(m)
    return maps


def load_trigger_mask(mask_path):
    """Load trigger mask from NPY or PNG."""
    mask_path = Path(mask_path)
    if mask_path.is_dir():
        # Try global_delta.npy as mask
        delta_path = mask_path / 'global_delta.npy'
        if delta_path.exists():
            delta = np.load(str(delta_path), allow_pickle=True)
            # Convert delta to binary mask
            if delta.ndim == 3 and delta.shape[0] in (1, 3):
                delta = delta.transpose(1, 2, 0)
            mask = (np.abs(delta).max(axis=-1) if delta.ndim == 3 else np.abs(delta)) > 1e-6
            return mask.astype(np.float32)
    elif mask_path.suffix == '.npy':
        return np.load(str(mask_path), allow_pickle=True).astype(np.float32)
    elif mask_path.suffix in ('.png', '.jpg'):
        m = plt.imread(str(mask_path))
        if m.max() > 1:
            m = m.astype(np.float32) / 255.0
        return m
    return None


def overlay_cam(image, cam, alpha=0.5):
    """Overlay CAM heatmap on image."""
    # Resize CAM to match image
    if cam.shape[:2] != image.shape[:2]:
        import cv2
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    # Normalize CAM
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Create heatmap
    cmap = plt.cm.inferno
    heatmap = cmap(cam_norm)[:, :, :3]
    # Blend
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    overlay = image * (1 - alpha) + heatmap * alpha
    return np.clip(overlay, 0, 1)


def compute_cam_trigger_overlap(cam, trigger_mask):
    """Compute overlap metrics between CAM and trigger mask."""
    if trigger_mask is None:
        return {}
    # Resize trigger to match CAM
    if trigger_mask.shape[:2] != cam.shape[:2]:
        import cv2
        trigger_mask = cv2.resize(trigger_mask, (cam.shape[1], cam.shape[0]))
    trigger_mask = (trigger_mask > 0.5).astype(np.float32)

    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_binary = (cam_norm > 0.5).astype(np.float32)  # top 50%

    # CAM-trigger overlap: fraction of high CAM within trigger region
    if trigger_mask.sum() > 0:
        cam_in_trigger = cam_norm[trigger_mask > 0.5].mean()
        cam_outside_trigger = cam_norm[trigger_mask < 0.5].mean()
        trigger_attention_ratio = cam_in_trigger / (cam_outside_trigger + 1e-8)
    else:
        cam_in_trigger = 0
        trigger_attention_ratio = 0

    # IoU
    intersection = (cam_binary * trigger_mask).sum()
    union = (cam_binary + trigger_mask).clip(0, 1).sum()
    iou = intersection / (union + 1e-8)

    return {
        'cam_in_trigger': float(cam_in_trigger),
        'cam_outside_trigger': float(cam_outside_trigger),
        'trigger_attention_ratio': float(trigger_attention_ratio),
        'iou': float(iou),
    }


def plot_cam_grid(images, saliency_maps, trigger_mask, image_names, args):
    """Create multi-panel CAM visualization figure."""
    n = len(images)
    has_mask = trigger_mask is not None
    n_cols = 5 if has_mask else 4
    # Layout: Image | Saliency | CAM Overlay | Trigger Mask (opt) | Overlap Map (opt)

    fig, axes = plt.subplots(n, n_cols, figsize=args.figsize or (n_cols * 1.5, n * 1.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Image', 'Grad-CAM', 'Overlay']
    if has_mask:
        col_titles.extend(['Trigger Mask', 'CAM∩Trigger'])
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, fontweight='bold')

    metrics_all = []
    for i in range(n):
        img = images[i]
        cam = saliency_maps[i] if i < len(saliency_maps) else np.zeros_like(img[:, :, 0] if img.ndim == 3 else img)

        # Col 1: Original image
        axes[i, 0].imshow(img)
        if image_names:
            axes[i, 0].set_ylabel(image_names[i], fontsize=8)
        axes[i, 0].axis('off')

        # Col 2: Saliency/CAM
        im = axes[i, 1].imshow(cam, cmap='inferno')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 3: Overlay
        overlay = overlay_cam(img, cam)
        axes[i, 2].imshow(overlay)
        axes[i, 2].axis('off')

        if has_mask:
            # Col 4: Trigger mask
            t_mask_disp = trigger_mask
            if t_mask_disp.shape[:2] != img.shape[:2]:
                import cv2
                t_mask_disp = cv2.resize(t_mask_disp, (img.shape[1], img.shape[0]))
            axes[i, 3].imshow(t_mask_disp, cmap='gray')
            axes[i, 3].axis('off')

            # Col 5: CAM ∩ Trigger overlap
            if t_mask_disp.shape[:2] != cam.shape[:2]:
                import cv2
                t_mask_cam = cv2.resize(t_mask_disp, (cam.shape[1], cam.shape[0]))
            else:
                t_mask_cam = t_mask_disp
            overlap_map = (cam / (cam.max() + 1e-8)) * (t_mask_cam > 0.5)
            axes[i, 4].imshow(overlap_map, cmap='hot')
            axes[i, 4].axis('off')

            metrics = compute_cam_trigger_overlap(cam, t_mask_cam)
            metrics_all.append(metrics)
            axes[i, 4].text(0.5, -0.1,
                            f'IoU={metrics["iou"]:.3f}\nRatio={metrics["trigger_attention_ratio"]:.1f}×',
                            transform=axes[i, 4].transAxes, ha='center', fontsize=6)

    fig.suptitle('Grad-CAM / Saliency Analysis — Clean vs Poisoned',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    basepath = Path(args.output) / args.prefix
    save_fig_all(fig, str(basepath), args.formats)

    if metrics_all and args.metadata:
        meta = {
            'n_samples': n,
            'overlap_metrics': metrics_all,
            'average_iou': float(np.mean([m['iou'] for m in metrics_all])) if metrics_all else 0,
            'average_trigger_attention_ratio': float(np.mean([m['trigger_attention_ratio'] for m in metrics_all])) if metrics_all else 0,
        }
        meta_path = Path(args.output) / f'{args.prefix}_metrics.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved metrics: {meta_path}")

    if args.caption:
        avg_iou = np.mean([m['iou'] for m in metrics_all]) if metrics_all else 0
        print(f"Figure X: **Grad-CAM analysis** of model attention on clean vs poisoned samples. "
              f"Average CAM-trigger overlap IoU = {avg_iou:.3f}. "
              f"High overlap indicates the trigger dominates model decisions.")

    return fig


def main():
    parser = get_parser("Saliency/CAM visualization for backdoor papers.")
    parser.add_argument('--images', required=True,
                        help='Directory of original images')
    parser.add_argument('--saliency', required=True,
                        help='Directory of saliency/CAM maps (.npy or .png)')
    parser.add_argument('--trigger-mask', default=None,
                        help='Trigger mask (.npy or FAAT trigger dir)')
    parser.add_argument('--n', type=int, default=4,
                        help='Number of samples (default: 4)')
    args = parser.parse_args()

    if not args.no_style:
        setup_paper_style()

    print(f"Loading images from: {args.images}")
    images, names = load_images(args.images, args.n)
    print(f"  Loaded {len(images)} images")

    print(f"Loading saliency maps from: {args.saliency}")
    saliency_maps = load_saliency_maps(args.saliency, args.n)
    print(f"  Loaded {len(saliency_maps)} maps")

    trigger_mask = None
    if args.trigger_mask:
        trigger_mask = load_trigger_mask(args.trigger_mask)
        if trigger_mask is not None:
            print(f"  Loaded trigger mask: {trigger_mask.shape}")

    if not images or not saliency_maps:
        print("Error: no images or saliency maps loaded")
        return

    plot_cam_grid(images, saliency_maps, trigger_mask, names, args)


if __name__ == '__main__':
    import pandas as pd
    main()
