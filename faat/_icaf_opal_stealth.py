"""Stealth probe for ICAF and OPAL triggers: L2/img, L_inf, SSIM (kornia), LPIPS (AlexNet) on
trigger-only (test-time) perturbation. Run after the artifact is saved.

Run: python -m faat._icaf_opal_stealth --trigger icaf --save_trigger resource/faat/icaf/a1.0_s1
     python -m faat._icaf_opal_stealth --trigger opal --save_trigger resource/faat/opal/b4_m8_l16_s1
"""
import argparse
import torch
from torchvision import datasets, transforms


def _metrics(x_clean, x_pert, lpips_fn, dev):
    diff = x_pert - x_clean
    l2 = diff.flatten(1).norm(dim=1).mean().item()
    linf = diff.abs().max().item()
    try:
        from kornia.metrics import ssim as kssim
        s = kssim(x_clean, x_pert, window_size=11).mean().item()
    except Exception:
        s = float('nan')
    with torch.no_grad():
        l = lpips_fn(x_clean * 2 - 1, x_pert * 2 - 1).mean().item()
    return l2, linf, s, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trigger', required=True, choices=['icaf', 'opal'])
    ap.add_argument('--save_trigger', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(dev).eval()

    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    idx = [i for i in range(len(ds)) if ds[i][1] != 0][:args.n]   # non-target test images
    x = torch.stack([ds[i][0] for i in idx]).to(dev)

    if args.trigger == 'icaf':
        from .icaf_trigger import load_icaf_mask, icaf_apply
        mask, alpha = load_icaf_mask(args.save_trigger, dev, size=32)
        m = mask()
        xp = icaf_apply(x, m, alpha)
        extra = 'alpha=%.3fpx' % alpha
    else:
        from .opal_trigger import load_opal_key, opal_project
        key, margin, block, linf = load_opal_key(args.save_trigger, dev)
        xp, _ = opal_project(x, key, margin, block=block, linf=linf)
        extra = 'block=%d margin=%.5f linf=%s' % (block, margin, linf)

    l2, linf, s, l = _metrics(x, xp, lpips_fn, dev)
    print('[%s %s] %s' % (args.trigger, args.save_trigger, extra))
    print('  TRIGGER-ONLY (test): L2/img=%.4f  L_inf=%.4f  SSIM=%.4f  LPIPS=%.5f' % (l2, linf, s, l))


if __name__ == '__main__':
    main()
