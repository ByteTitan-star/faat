"""Stage A rule-based adaptive trigger: deterministic image-space rule.

s_i (image stats) -> (DCT band weights, L2 budget eps) -> delta_adaptive_i.

Design choices (Stage A, intentionally simple; Stage B replaces this with a learned
differentiable policy + feature alignment):
  * Emphasize low-mid frequency bands -> robust to the RandomCrop/Flip augmentation
    applied AFTER injection (host-repo constraint C3). High bands are discouraged.
  * More texture -> slightly more mid-high contribution + larger eps (the image can
    hide more there). Smooth images -> tight low-freq, small eps.
  * delta_adaptive is normalized to L2 = eps, so the total distortion is bounded and
    comparable to baselines.
"""
import numpy as np
import torch

from dct import dct_2d, idct_2d
from .image_stats import texture_complexity, band_energy, band_weight_map, N_BANDS

# low-mid emphasis (robust to crop/flip); high bands suppressed
_BASE_BANDS = np.array([0.0, 1.0, 1.0, 0.6, 0.3, 0.1, 0.0, 0.0])
# extra mid-high allowed as texture rises (still modest)
_TEX_EXTRA = np.array([0.0, 0.0, 0.0, 0.2, 0.3, 0.3, 0.2, 0.1])


def rule_params(img):
    """img: [3,H,W] -> (band_weights [N_BANDS] summing to 1, eps float in ~[0.01,0.04])."""
    tex = texture_complexity(img)
    # texture -> [0,1] via sigmoid. Calibrated on CIFAR-10 (median tex≈0.066, IQR≈0.041);
    # RECALIBRATE per dataset (run faat/calibrate_texture.py).
    t = 1.0 / (1.0 + np.exp(-(tex - 0.066) / 0.041))
    band_weights = _BASE_BANDS + _TEX_EXTRA * t
    band_weights = band_weights / (band_weights.sum() + 1e-12)
    eps = 0.01 + 0.03 * float(t)  # L2 budget for the adaptive residual
    return band_weights, eps


def generate_adaptive_delta(img, band_weights, eps, seed=0):
    """Produce a DCT band-passed noise delta with L2 norm == eps.

    img: [3,H,W] (only shape is used). Returns [3,H,W] float tensor.
    """
    size = img.shape[-1]
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn((3, size, size), generator=g)
    wmap = band_weight_map(band_weights, size=size)  # [H,W]
    D = dct_2d(noise.unsqueeze(0)).squeeze(0) * wmap  # [3,H,W]
    delta = idct_2d(D)
    norm = delta.norm(p=2) + 1e-12
    return delta * (eps / norm)
