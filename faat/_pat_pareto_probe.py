"""Probe SSIM/LPIPS/L_inf for all existing PAT triggers -> map the ASR-SSIM Pareto.
Target: find alphas with SSIM>=0.97 (candidate to dominate BppAttack's our-measured 0.97)."""
import torch
from torchvision import datasets, transforms

from faat.pat_trigger import load_pat_delta, pat_apply
from faat._icaf_opal_stealth import _metrics

DEV = 'cuda'
import lpips
lpips_fn = lpips.LPIPS(net='alex').to(DEV).eval()

ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
idx = [i for i in range(len(ds)) if ds[i][1] != 0][:500]
x = torch.stack([ds[i][0] for i in idx]).to(DEV)

print('alpha | SSIM | LPIPS | L_inf | L2/img')
print('-' * 50)
for a in [1.5, 1.0, 0.75, 0.5, 0.3, 0.2]:
    p = 'resource/faat/pat/a%s/pat.pth' % a
    import os
    if not os.path.exists(p):
        continue
    delta, alpha = load_pat_delta('resource/faat/pat/a%s' % a, DEV)
    xp = pat_apply(x, delta, alpha)
    l2, linf, s, l = _metrics(x, xp, lpips_fn, DEV)
    flag = '  <-- SSIM>=0.97' if s >= 0.97 else ''
    print('%.2f  | %.4f | %.5f | %.4f | %.4f%s' % (alpha, s, l, linf, l2, flag))
