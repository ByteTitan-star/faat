"""Sweep (scale, ksize) for a STEALTHY+EFFECTIVE RKT config: proxyASR + SSIM + L2.
No-gain, hard sum=1 kernel (brightness-preserving resampling). Goal: find a config with
proxyASR>=0.7 AND SSIM>=0.9 before committing to a full victim run. If none exists, RKT hits
the low-amplitude-transform wall (honest negative).

Run: CUDA_VISIBLE_DEVICES=0 python -m faat._rkt_sweep --scales 0.4,0.5,0.6,0.7 --ksizes 5,7
"""
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.rkt_trigger import optimize_rkt_trigger

DEV = 'cuda'


def ssim_batch(x, y, win=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    w = torch.ones(3, 1, win, win, device=x.device) / (win * win)

    def box(t):
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode='reflect'), w, groups=3)

    mx, my = box(x), box(y)
    vx = box(x * x) - mx * mx
    vy = box(y * y) - my * my
    cxy = box(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean().item()


def load_proxy(path, nc=10):
    ck = torch.load(path, map_location=DEV)
    m = ResNet18(num_classes=nc).to(DEV)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='0.4,0.5,0.6,0.7')
    ap.add_argument('--ksizes', default='5,7')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--stealth_lam', type=float, default=1e-2)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--proxy', default='resource/faat/proxy/resnet18_clean_cifar10.pth')
    args = ap.parse_args()

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)
    proxy = load_proxy(args.proxy)
    xs = imgs[:1000]

    print('%-6s %-6s %10s %10s %10s' % ('scale', 'ksize', 'proxyASR', 'SSIM', 'L2'))
    for s in [float(v) for v in args.scales.split(',')]:
        for k in [int(v) for v in args.ksizes.split(',')]:
            info = optimize_rkt_trigger(proxy, imgs, args.y_target, DEV, scale=s, ksize=k,
                                        steps=args.steps, stealth_lam=args.stealth_lam,
                                        seed=1, log_every=9999)
            trig = info['trig']
            with torch.no_grad():
                xp = trig(xs)
                ssim = ssim_batch(xs, xp)
                l2 = (xp - xs).flatten(1).norm(dim=1).mean().item()
            print('%-6.2f %-6d %10.3f %10.3f %10.3f' % (s, k, info['final_proxy_asr'], ssim, l2))


if __name__ == '__main__':
    main()
