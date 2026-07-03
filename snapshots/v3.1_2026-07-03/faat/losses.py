"""FAAT Stage B losses (all differentiable, zero external dependency).

  * align_loss  : ||f(x') - c_target||_2  (THE core C-mechanism of FAAT).
  * ssim / ssim_loss : differentiable structural similarity (self-written, no kornia/lpips).
  * l2_loss     : mean per-sample L2 of the perturbation (stealth magnitude).
  * freq_loss   : penalises high-frequency DCT energy of the perturbation
                  -> keeps delta low/mid-band, robust to the RandomCrop/Flip
                     augmentation applied AFTER injection (host-repo constraint C3).

All inputs are images / deltas shaped [B,3,H,W] in [0,1]; dct_2d is scale-agnostic
and differentiable, so these compose cleanly into the offline optimisation graph.
"""
import torch
import torch.nn.functional as F

from dct import dct_2d


# --------------------------------------------------------------------------- #
# Feature alignment (core)
# --------------------------------------------------------------------------- #
def align_loss(features, c_target):
    """features: [B,D]; c_target: [D]. Returns mean per-sample L2 distance."""
    return (features - c_target).norm(dim=1).mean()


# --------------------------------------------------------------------------- #
# Differentiable SSIM (self-written, Gaussian window)
# --------------------------------------------------------------------------- #
def _gauss_window(size=11, sigma=1.5, channels=3):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim(x, y, size=11, sigma=1.5, data_range=1.0):
    """Differentiable SSIM. x,y: [B,C,H,W] in [0,1]. Returns scalar mean in [-1,1]."""
    C = x.shape[1]
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = _gauss_window(size, sigma, C).to(device=x.device, dtype=x.dtype)
    pad = size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


def ssim_loss(x, y, **kw):
    """Perceptual stealth loss = 1 - SSIM (0 when identical, up to 2 when inverted)."""
    return 1.0 - ssim(x, y, **kw)


# --------------------------------------------------------------------------- #
# Magnitude / frequency stealth
# --------------------------------------------------------------------------- #
def l2_loss(delta):
    """delta: [B,3,H,W] -> mean per-sample L2 norm."""
    return delta.flatten(1).norm(dim=1).mean()


def freq_loss(delta, band_idx, n_bands=8, high_frac=0.5):
    """Penalise DCT energy in the upper frequency bands of `delta`.

    band_idx: [H,W] long tensor of radial-band ids (shared with the generator).
    Returns mean over batch of summed high-band DCT energy.
    """
    D = dct_2d(delta)                       # [B,3,H,W]
    E = (D ** 2).sum(dim=1)                 # [B,H,W]
    high_mask = (band_idx >= int(n_bands * high_frac)).to(dtype=E.dtype)
    return (E * high_mask.unsqueeze(0)).sum(dim=(1, 2)).mean()
