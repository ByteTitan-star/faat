"""Stealth (imperceptibility) metrics. Pure torch, no extra dependencies.

All inputs are image tensors shaped [B, 3, H, W] in [0, 1].
"""
import torch
import torch.nn.functional as F

try:
    from dct import dct_2d
except Exception:  # pragma: no cover - allow import without repo root on path
    dct_2d = None


def _gaussian_window(window_size=11, sigma=1.5, channels=3):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(x, y, window_size=11, sigma=1.5, data_range=1.0, size_average=True):
    """Differentiable SSIM. Returns per-image mean if size_average=False."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ch = x.shape[1]
    window = _gaussian_window(window_size, sigma, ch).to(x.device).type_as(x)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=ch)
    mu_y = F.conv2d(y, window, padding=pad, groups=ch)
    mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=ch) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=ch) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=ch) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])


def l2(x, y):
    """Per-image L2 distance -> [B]."""
    return torch.sqrt(((x - y) ** 2).sum(dim=[1, 2, 3]) + 1e-12)


def linf(x, y):
    """Per-image L_inf distance -> [B]."""
    return (x - y).abs().reshape(x.shape[0], -1).max(dim=1)[0]


def dct_l1(x, y):
    """Per-image mean |DCT(x) - DCT(y)| -> [B]. Measures spectral detectability."""
    if dct_2d is None:
        raise RuntimeError("dct_2d unavailable; run from repo root or add it to sys.path")
    return (dct_2d(x) - dct_2d(y)).abs().reshape(x.shape[0], -1).mean(dim=1)


def stealth_metrics(x, x_adv):
    """Batch-mean stealth metrics between clean x and adversarial x_adv ([0,1])."""
    with torch.no_grad():
        out = {
            'L2': l2(x, x_adv).mean().item(),
            'Linf': linf(x, x_adv).mean().item(),
            'SSIM': ssim(x, x_adv).item(),
        }
        if dct_2d is not None:
            out['DCT_L1'] = dct_l1(x, x_adv).mean().item()
        return out
