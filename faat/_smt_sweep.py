"""SMT sweep: spectral_weight vs proxy-ASR + stealth (SSIM/Linf). Compare to Narcissus (w=0)."""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from faat.proxy import load_proxy
from faat.smt_trigger import optimize_smt_trigger

DEV = 'cuda'


def ssim_torch(x, y, win=11):
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
    print('%-16s %8s %8s %10s %10s %8s' % ('spec_weight', 'proxyASR', 'SSIM', 'Linf/255', 'meanD/255', 'L2'))
    for w in [0.0, 0.0005, 0.002, 0.008, 0.03]:
        delta, info = optimize_smt_trigger(proxy, imgs, 0, DEV, l2_budget=1.5, steps=2000,
                                           lr=0.02, spectral_weight=w, batch_size=128, log_every=10000)
        d = torch.from_numpy(delta).to(DEV)
        x = imgs[:1000]
        xp = torch.clamp(x + d, 0, 1)
        diff = xp - x
        print('%-16s %8.3f %8.3f %10.1f %10.2f %8.3f' % (
            w, info['final_proxy_asr'], ssim_torch(x, xp),
            diff.abs().max().item() * 255, diff.abs().mean().item() * 255,
            diff.flatten(1).norm(dim=1).mean().item()))


if __name__ == '__main__':
    main()
