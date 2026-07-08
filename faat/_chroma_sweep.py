"""ChromaTrigger LUT-only sweep: find the ASR knee vs LUT deviation, measure stealth.
Pure global colour remap (pattern disabled) -- the cleanest 'invisible trigger form'.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from faat.proxy import load_proxy
from faat.chroma_trigger import optimize_chroma_trigger, apply_lut

DEV = 'cuda'


def ssim_torch(x, y, win=11):
    """Averaged-window SSIM over [N,3,H,W] in [0,1]. Returns mean scalar."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    w = torch.ones(3, 1, win, win, device=x.device) / (win * win)

    def box(t):
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode='reflect'), w, groups=3)

    mu_x, mu_y = box(x), box(y)
    var_x = box(x * x) - mu_x * mu_x
    var_y = box(y * y) - mu_y * mu_y
    cov_xy = box(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return ssim.mean().item()


def main():
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(8000)]).to(DEV)
    proxy = load_proxy('resource/faat/proxy/resnet18_clean_cifar10.pth', 10, DEV)

    print('%-14s %8s %8s %10s %10s %8s' % ('lut_max_dev', 'proxyASR', 'SSIM', 'Linf/255', 'meanD/255', 'L2'))
    for lutdev in [0.06, 0.10, 0.15, 0.20, 0.25, 0.30]:
        res = optimize_chroma_trigger(
            proxy, imgs, 0, DEV, l2_budget=1.5, steps=1500, lr=0.02,
            K=16, n_basis=48, lambda_lut=0.05, lut_max_dev=lutdev,
            batch_size=128, log_every=10000)  # quiet
        V = torch.from_numpy(res['V']).to(DEV)
        P = torch.from_numpy(res['P']).to(DEV)
        x = imgs[:1000]
        xp = torch.clamp(apply_lut(x, V) + P, 0, 1)
        d = xp - x
        print('%-14s %8.3f %8.3f %10.1f %10.2f %8.3f' % (
            lutdev, res['final_proxy_asr'], ssim_torch(x, xp),
            d.abs().max().item() * 255, d.abs().mean().item() * 255,
            d.flatten(1).norm(dim=1).mean().item()))


if __name__ == '__main__':
    main()
