"""Self-contained Narcissus-style universal trigger generator (v4 core).

WHY THIS EXISTS
---------------
v3.1's strength came from the authors' pre-computed ``noise_01000.pth``, which is
CIFAR-10-specific (3x32x32, 10-class). It does NOT transfer: that is exactly why
GTSRB collapsed to ASR 0.9% -- with no usable seed direction, the CE-proxy path in
optimize.py started from 0.02-scale noise and 2000 coupled steps could not climb to
a strong universal trigger.

This module removes the dependency entirely. It optimises a SINGLE universal
perturbation ``delta_global`` [3,H,W] FROM SCRATCH against the *clean proxy* so that
the proxy classifies ``x + delta_global`` as ``target`` for as many clean images as
possible, under an L2 (stealth) budget. This is precisely the Narcissus objective,
run by us on whatever dataset the proxy was trained on -> it generalises to GTSRB,
Tiny-ImageNet, CIFAR-100, ... without any authors' artifact.

The produced ``delta_global`` then seeds the bounded-adaptive FAAT pipeline
(optimize.py with ``--global_mode from_scratch --fix_global``). Everything
downstream (bounded adaptive, L_align, injection, training, defenses) is reused
unchanged -- only the *source* of delta_global changes.

CLI sanity check (no victim training; just "does the clean proxy get fooled?"):
  python -m faat.global_trigger --dataset gtsrb --data_dir data/GTSRB32 \
      --num_classes 43 --y_target 0 \
      --proxy_path resource/faat/proxy/resnet18_clean_gtsrb.pth \
      --l2_budget 1.5 --steps 8000 --device cuda \
      --save_dir resource/faat/v4/gtsrb/self_gen
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils import set_random_seed
from .proxy import load_proxy


# --------------------------------------------------------------------------- #
# Clean-image cache (GPU tensor) -- one universal trigger optimises over ALL
# clean images, so caching them on-device makes thousands of steps cheap.
# --------------------------------------------------------------------------- #
def build_clean_image_tensor(dataset, size, device, cap=60000):
    """Stack clean images into a [M,3,size,size] float tensor on `device`.

    `dataset` is a torchvision-style dataset whose dataset[i][0] is a tensor image.
    Capped at `cap` (random sub-sample) to bound GPU memory on larger datasets.
    """
    n = len(dataset)
    if n > cap:
        rng = np.random.RandomState(0)
        keep = sorted(rng.choice(n, cap, replace=False).tolist())
    else:
        keep = list(range(n))
    imgs = [dataset[i][0] for i in keep]
    x = torch.stack(imgs).float().to(device)        # [M,3,H,W]
    if x.shape[-1] != size or x.shape[-2] != size:
        import torch.nn.functional as Fn
        x = Fn.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    return x


@torch.no_grad()
def proxy_argmax(proxy, images, device, batch_size=512):
    """Predicted labels for every image in `images`, chunked (full-set forward
    in one shot blows up activations / trips cuDNN on big sets)."""
    out = []
    for s in range(0, images.shape[0], batch_size):
        out.append(proxy(images[s:s + batch_size]).argmax(1))
    return torch.cat(out)


@torch.no_grad()
def proxy_asr(proxy, images, delta, target, device, eval_n=3000, batch_size=512):
    """Fraction of clean images the proxy classifies as `target` after +delta.

    (Higher proxy-ASR correlates with higher victim test-ASR -- Narcissus mechanism.)
    """
    M = images.shape[0]
    n = min(eval_n, M)
    idx = torch.randperm(M, device=device)[:n]
    d = delta.reshape(1, *delta.shape) if delta.dim() == 3 else delta
    correct = 0
    for s in range(0, n, batch_size):
        sl = idx[s:s + batch_size]
        out = proxy(torch.clamp(images[sl] + d, 0.0, 1.0))
        correct += out.argmax(1).eq(target).sum().item()
    return correct / n


# --------------------------------------------------------------------------- #
# Narcissus-from-scratch engine
# --------------------------------------------------------------------------- #
def optimize_global_trigger(proxy, images, target, device,
                            l2_budget=1.5, steps=8000, lr=0.02,
                            batch_size=128, loss='ce', margin=10.0,
                            init='zero', seed=0, log_every=200,
                            exclude_target=True, logger=print):
    """Optimise a single universal trigger ``delta`` [3,H,W] from scratch.

    Objective (Narcissus): make the frozen clean proxy predict `target` for
    ``clamp(x + delta, 0, 1)`` over clean images, under a hard L2 budget
    ``l2_budget`` (the stealth radius). Returns (delta_numpy[3,H,W], info_dict).

    loss:
      'ce'  -- minimise CE(proxy(x+delta), target)             (authors' choice)
      'cw'  -- targeted Carlini-Wagner margin: push target logit above the
               best other by `margin` (often more universal / transferable)
    init:
      'zero'   -- start at 0 (Narcissus default; the L2 projection bounds growth)
      'rand'   -- tiny Gaussian
    exclude_target:
      drop images already in the target class (they trivially classify as target
      and add no gradient signal toward a *cross-class* universal direction).
    """
    set_random_seed(seed)
    H, W = images.shape[-2], images.shape[-1]

    # operate only on non-target clean images (the hard, informative set)
    if exclude_target:
        lbl = proxy_argmax(proxy, images, device)
        pool = torch.where(lbl != target)[0]
        if pool.numel() < batch_size:           # degenerate fallback
            pool = torch.arange(images.shape[0], device=device)
    else:
        pool = torch.arange(images.shape[0], device=device)
    pool = pool.to(device)
    M = pool.numel()
    logger('[gtrig] optimising universal trigger: %d clean non-target images, '
           'L2 budget=%.3f, steps=%d, loss=%s, lr=%.3f' % (M, l2_budget, steps, loss, lr))

    if init == 'zero':
        delta = torch.zeros(3, H, W, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(int(seed) + 7)
        delta = 0.01 * torch.randn(3, H, W, generator=g, device=device)
    delta = delta.contiguous().requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)
    t0 = time.time()
    history = []
    for step in range(steps):
        bi = pool[torch.randint(0, M, (min(batch_size, M),), device=device)]
        x = images[bi]
        logits = proxy(torch.clamp(x + delta.unsqueeze(0), 0.0, 1.0))
        if loss == 'ce':
            tgt = torch.full((x.shape[0],), target, dtype=torch.long, device=device)
            L = F.cross_entropy(logits, tgt)
        else:  # targeted CW margin
            tg = logits[:, target]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, target] = False
            other = logits[mask].view(x.shape[0], -1).max(1).values
            L = F.relu(margin - (tg - other)).mean()
        opt.zero_grad(); L.backward(); opt.step()

        # hard L2 projection (shrink only) -- precise stealth budget control
        with torch.no_grad():
            nrm = delta.norm()
            if nrm.item() > l2_budget:
                delta.data.mul_(l2_budget / (nrm + 1e-12))

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                asr = proxy_asr(proxy, images, delta.detach(), target, device)
            rec = (step, float(L.item()), float(delta.norm().item()), float(asr),
                   time.time() - t0)
            history.append(rec)
            logger('[gtrig %5d] L=%.4f |delta|=%.3f proxyASR=%.3f (%.0fs)' % rec)

    delta_np = delta.detach().cpu().numpy()
    info = {'l2_budget': l2_budget, 'steps': steps, 'loss': loss, 'lr': lr,
            'init': init, 'final_l2': float(delta.norm().item()),
            'final_proxy_asr': float(proxy_asr(proxy, images, delta.detach(),
                                               target, device)),
            'history': history}
    return delta_np, info


# --------------------------------------------------------------------------- #
# CLI -- standalone sanity check (no victim training)
# --------------------------------------------------------------------------- #
def _load_dataset(args):
    if args.dataset == 'cifar10':
        return datasets.CIFAR10(root='./data', train=True,
                                transform=transforms.ToTensor(), download=True)
    if args.dataset == 'cifar100':
        return datasets.CIFAR100(root='./data100', train=True,
                                 transform=transforms.ToTensor(), download=True)
    tf = transforms.Compose([transforms.Resize(args.size), transforms.ToTensor()])
    return datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=tf)


def main():
    ap = argparse.ArgumentParser('Self-contained Narcissus trigger generator (v4)')
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--proxy_path', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--l2_budget', type=float, default=1.5)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--loss', choices=['ce', 'cw'], default='ce')
    ap.add_argument('--margin', type=float, default=10.0)
    ap.add_argument('--init', choices=['zero', 'rand'], default='zero')
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--save_dir', default='./resource/faat/v4/self_gen')
    args = ap.parse_args()

    set_random_seed(args.seed)
    proxy = load_proxy(args.proxy_path, args.num_classes, args.device)
    ds = _load_dataset(args)
    images = build_clean_image_tensor(ds, args.size, args.device)
    print('[gtrig] dataset=%s images=%d target=%d proxy=%s' %
          (args.dataset, images.shape[0], args.y_target, args.proxy_path))

    delta_np, info = optimize_global_trigger(
        proxy, images, args.y_target, args.device,
        l2_budget=args.l2_budget, steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, loss=args.loss, margin=args.margin,
        init=args.init, seed=args.seed, log_every=args.log_every)

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'global_delta.npy'), delta_np)
    meta = dict(vars(args))
    meta.update({k: v for k, v in info.items() if k != 'history'})
    with open(os.path.join(args.save_dir, 'global_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('[gtrig] DONE -> %s  final L2=%.3f proxyASR=%.3f' %
          (args.save_dir, info['final_l2'], info['final_proxy_asr']))


if __name__ == '__main__':
    main()
