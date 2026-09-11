"""Sweep (tile, levels) for a STEALTHY+EFFECTIVE dither config: proxyASR + SSIM + L2.
Goal: find proxyASR>=0.7 AND SSIM>=0.9 (the config RKT could NOT reach). Success -> dither is
a viable structure-preserving trigger -> proceed to victim training. Failure -> trigger path
is dead on CIFAR (5+ mechanisms walled).

Run: CUDA_VISIBLE_DEVICES=0 python -m faat._dither_sweep --tiles 2,4,8 --levelss 8,16,32
"""
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.dither_trigger import optimize_dither_trigger

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


def bpp_ssim_ref(xs, levels=(24, 28, 8)):
    """BppAttack-style uniform per-channel quantization SSIM (reference point)."""
    out = xs.clone()
    for ch, L in enumerate(levels):
        step = 1.0 / L
        out[:, ch] = torch.round(out[:, ch] / step) * step
    return ssim_batch(xs, torch.clamp(out, 0, 1))


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
    ap.add_argument('--tiles', default='2,4,8')
    ap.add_argument('--levels', default='8,16,32')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--proxy', default='resource/faat/proxy/resnet18_clean_cifar10.pth')
    args = ap.parse_args()

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)
    proxy = load_proxy(args.proxy)
    xs = imgs[:1000]
    print('[ref] BppAttack(24:28:8) uniform-quant SSIM = %.3f' % bpp_ssim_ref(xs))
    print('%-5s %-7s %10s %10s %10s' % ('tile', 'levels', 'proxyASR', 'SSIM', 'L2'))
    hits = []
    for t in [int(v) for v in args.tiles.split(',')]:
        for L in [int(v) for v in args.levels.split(',')]:
            info = optimize_dither_trigger(proxy, imgs, args.y_target, DEV, tile=t, levels=L,
                                           steps=args.steps, seed=1, log_every=9999)
            trig = info['trig']
            with torch.no_grad():
                xp = trig(xs)
                ssim = ssim_batch(xs, xp)
                l2 = (xp - xs).flatten(1).norm(dim=1).mean().item()
            star = '  <-- STEALTHY+EFF' if (ssim >= 0.9 and info['final_proxy_asr'] >= 0.7) else ''
            if star:
                hits.append((t, L, info['final_proxy_asr'], ssim))
            print('%-5d %-7d %10.3f %10.3f %10.3f%s' % (t, L, info['final_proxy_asr'], ssim, l2, star))
    print('\n=== configs with SSIM>=0.9 AND proxyASR>=0.7: %d ===' % len(hits))
    for h in hits:
        print('  tile=%d levels=%d proxyASR=%.3f SSIM=%.3f' % h)


if __name__ == '__main__':
    main()
