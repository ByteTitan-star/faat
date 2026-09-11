"""Per-sample image-space statistics for Stage A rule-based FAAT (no proxy needed).

These are an *image-space proxy* for the feature alignment that arrives in Stage B
(where a clean ResNet proxy supplies c_target and L_align). Here we only use cheap,
deterministic, GPU-free image statistics: texture complexity (Laplacian variance) and
DCT radial-band energy.
"""
import numpy as np
import torch
import torch.nn.functional as F

from dct import dct_2d

N_BANDS = 8


def _laplacian_kernel():
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return k.view(1, 1, 3, 3)


def texture_complexity(img):
    """img: [3,H,W] in [0,1] -> scalar Laplacian variance (higher = more textured)."""
    gray = img.mean(dim=0, keepdim=True).unsqueeze(0).float()  # [1,1,H,W]
    k = _laplacian_kernel()
    lap = F.conv2d(gray, k, padding=1)
    return float(lap.var().item())


def _band_index_map(size=32, n_bands=N_BANDS):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size))
    radius = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    rmax = radius.max().item()
    edges = torch.linspace(0, rmax + 1e-6, n_bands + 1)
    band_idx = torch.bucketize(radius, edges[1:-1])  # 0..n_bands-1
    return band_idx


def band_energy(img, n_bands=N_BANDS, size=32):
    """img: [3,H,W] -> [n_bands] normalized DCT radial-band energy fractions (sum=1)."""
    x = img.unsqueeze(0)  # [1,3,H,W]
    D = dct_2d(x.float())
    E = (D[0] ** 2).sum(dim=0)  # [H,W] per-coeff energy summed over channels
    band_idx = _band_index_map(size, n_bands)
    energies = torch.zeros(n_bands)
    for b in range(n_bands):
        energies[b] = E[band_idx == b].sum()
    energies = energies / (energies.sum() + 1e-12)
    return energies.numpy().astype(np.float64)


def band_weight_map(band_weights, size=32, n_bands=N_BANDS):
    """band_weights: [n_bands] -> [size,size] per-pixel weight by radial band."""
    band_idx = _band_index_map(size, n_bands)
    wmap = torch.zeros(size, size)
    for b in range(n_bands):
        wmap[band_idx == b] = float(band_weights[b])
    return wmap
