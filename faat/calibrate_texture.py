"""Calibrate the Stage A texture rule on real CIFAR-10 (CPU, no GPU).

Reports the Laplacian-variance distribution and a suggested sigmoid threshold/slope
centered on the real median, so delta_adaptive actually discriminates samples instead
of saturating. Run: CUDA_VISIBLE_DEVICES="" python faat/calibrate_texture.py
"""
import os
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from faat.image_stats import texture_complexity, band_energy  # noqa: E402


def t_of(thr, slope, tex):
    return 1.0 / (1.0 + np.exp(-(tex - thr) / slope))


def main(n=1000):
    torch.manual_seed(0)
    ds = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=False)
    vals, bands = [], []
    for i in range(0, min(n * 5, len(ds)), 5):
        vals.append(texture_complexity(ds[i][0]))
        if len(bands) < 200:
            bands.append(band_energy(ds[i][0]))
    vals = np.array(vals)
    bands = np.array(bands)
    p25, med, p75 = np.percentile(vals, [25, 50, 75])

    print('TEXTURE Laplacian-var  min=%.5f p25=%.5f median=%.5f p75=%.5f max=%.5f'
          % (vals.min(), p25, med, p75, vals.max()))
    print('current rule (thr=0.02,slope=0.01): t median=%.3f IQR=[%.3f,%.3f]'
          % (t_of(0.02, 0.01, med), t_of(0.02, 0.01, p25), t_of(0.02, 0.01, p75)))

    iqr = max(p75 - p25, 1e-5)
    print('SUGGESTED (thr=median=%.5f, slope=iqr=%.5f): t IQR=[%.3f,%.3f]'
          % (med, iqr, t_of(med, iqr, p25), t_of(med, iqr, p75)))
    print('BAND energy (8 bands) mean over samples:', np.round(bands.mean(0), 3).tolist())


if __name__ == '__main__':
    main()
