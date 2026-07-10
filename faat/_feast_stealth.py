"""FEAST stealth probe: measure trigger-only (test-time) and poison (train-time) stealth of a
saved FEAST artifact. Reports L2/img, L_inf, SSIM (kornia), LPIPS (AlexNet).

Run: python -m faat._feast_stealth --save_trigger resource/faat/feast/pix_b1.0_starve_s1 \
       --starve_eps 0.03137 --n 500
"""
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from .feast_trigger import load_feast_artifact, feast_apply
from .feast_starve import starve_batch


def _metrics(x_clean, x_pert, lpips_fn, device):
    diff = x_pert - x_clean
    l2 = diff.flatten(1).norm(dim=1).mean().item()
    linf = diff.abs().max().item()
    try:
        from kornia.metrics import ssim as kssim
        s = kssim(x_clean, x_pert, window_size=11).mean().item()
    except Exception:
        s = float('nan')
    with torch.no_grad():
        # lpips expects [-1,1], [B,3,H,W]
        l = lpips_fn(x_clean * 2 - 1, x_pert * 2 - 1).mean().item()
    return l2, linf, s, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_trigger', required=True)
    ap.add_argument('--starve_eps', type=float, default=0.0)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(dev).eval()

    from cifar_resnet import ResNet18
    ck = torch.load('resource/faat/proxy/resnet18_clean_cifar10.pth', map_location=dev)
    proxy = ResNet18(num_classes=10).to(dev); proxy.load_state_dict(ck['state_dict']); proxy.eval()
    for p in proxy.parameters():
        p.requires_grad_(False)

    mode, trig = load_feast_artifact(args.save_trigger, dev)
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    # sample target-class images (the poison candidates)
    idx = [i for i, (_, l) in enumerate(ds) if l == args.y_target][:args.n]
    x = torch.stack([ds[i][0] for i in idx]).to(dev)

    # trigger-only (test-time): clean -> clean + trigger
    xt = feast_apply(x, mode, trig)
    l2t, linft, st, lt = _metrics(x, xt, lpips_fn, dev)

    # poison (train-time): clean -> starve(clean) + trigger
    if args.starve_eps > 0:
        xs = starve_batch(x, proxy, args.y_target, args.starve_eps, steps=30, device=dev)
        xp = feast_apply(xs, mode, trig)
        l2p, linfp, sp, lp = _metrics(x, xp, lpips_fn, dev)
    else:
        l2p = linfp = sp = lp = float('nan')

    print('[%s mode=%s eps=%.4f]' % (args.save_trigger, mode, args.starve_eps))
    print('  TRIGGER-ONLY (test):  L2/img=%.4f  L_inf=%.4f  SSIM=%.4f  LPIPS=%.5f' % (l2t, linft, st, lt))
    print('  POISON (train):       L2/img=%.4f  L_inf=%.4f  SSIM=%.4f  LPIPS=%.5f' % (l2p, linfp, sp, lp))


if __name__ == '__main__':
    main()
