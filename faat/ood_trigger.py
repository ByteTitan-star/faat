"""OOD-calibrated universal trigger -- A/B/C mechanism-isolation engine (48h core).

Scientific question (CLAUDE.md): can arbitrary external OOD samples, used as an
EXPLICIT negative calibration reference during trigger learning, produce more
target-specific (better-learnable) clean-label triggers than positive-only
target alignment -- and than merely having OOD as extra data?

Arms -- exactly ONE variable differs between adjacent arms:

  mode 'a'   pool = TARGET-CLASS clean images    objective = CE->target
             (positive-only target alignment; OOD never enters)
  mode 'b'   pool = TARGET + OOD images          objective = CE->target
             (OOD enters ONLY as extra optimisation data -- "surrogate/resource" role)
  mode 'c'   pool = TARGET-CLASS clean images    objective = CE->target + w * calib(OOD)
             (OOD enters ONLY as the explicit negative calibration reference)
  mode 'cur' pool = NON-TARGET clean images      objective = CE->target
             (frozen-baseline from_scratch behaviour -- reference point)

  A vs B isolates "more optimisation data";  A vs C isolates "the calibration term";
  B vs C isolates the FUNCTIONAL ROLE of OOD (data vs reference).

Calibration forms (--calib):
  cos  hinge on feature alignment:  mean_z relu( cos(f(z+delta), c_target) )
       -- delta must not rotate OOD features toward the target anchor (cos<=0 free)
  prob logit-level:                 mean_z softmax(proxy(z+delta))[target]
       -- delta must not raise P(target) on OOD inputs

Diagnostics logged every log_every steps: proxyASR separately on target-class /
non-target / OOD pools. The specificity signature of mode 'c' is
ASR(target) high while ASR(ood) stays low -- visible at trigger stage, before
any victim training.

Output (global_delta.npy + global_meta.json) is drop-in compatible with
optimize.py / apply_trigger.py; the FAAT downstream (adaptive residual, L_align,
injection, victim training) is reused unchanged. Only the SOURCE of delta_global
differs between arms.

Standalone CLI:
  python -m faat.ood_trigger --mode c --dataset cifar10 --ood_dataset cifar100 \
      --proxy_path resource/faat/proxy/resnet18_clean_cifar10.pth \
      --l2_budget 1.5 --steps 8000 --save_dir resource_ood/triggers/c_cifar10
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
from .global_trigger import build_clean_image_tensor, proxy_asr


# --------------------------------------------------------------------------- #
# Image+label tensors / OOD loading
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_image_and_label_tensors(dataset, size, device, cap=60000):
    """[M,3,H,W] float images + [M] long labels on `device` (random cap-subsample)."""
    n = len(dataset)
    if n > cap:
        rng = np.random.RandomState(0)
        keep = sorted(rng.choice(n, cap, replace=False).tolist())
    else:
        keep = list(range(n))
    imgs, lbls = zip(*[dataset[i] for i in keep])
    x = torch.stack(imgs).float().to(device)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    y = torch.tensor([int(l) for l in lbls], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def load_ood_images(name, data_dir, size, device, cap=20000):
    """Arbitrary EXTERNAL OOD pool for calibration (nothing from the victim task).

    name: cifar10 | cifar100 | tiny | imagefolder(data_dir). Capped at `cap`.
    """
    if name == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=True,
                              transform=transforms.ToTensor(), download=True)
    elif name == 'cifar100':
        ds = datasets.CIFAR100(root='./data100', train=True,
                               transform=transforms.ToTensor(), download=True)
    elif name == 'tiny':
        ds = datasets.ImageFolder(root=os.path.join('./data_tiny', 'train'),
                                  transform=transforms.ToTensor())
    elif name == 'imagefolder':
        ds = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
    else:
        raise ValueError('unknown ood_dataset: %s' % name)
    x, _ = build_image_and_label_tensors(ds, size, device, cap=cap)
    return x


@torch.no_grad()
def _feature_mean(proxy, imgs, device, batch_size=256):
    """Mean penultimate feature of `imgs` (c_target anchor)."""
    feats = []
    for s in range(0, imgs.shape[0], batch_size):
        feats.append(proxy.extract_feature(imgs[s:s + batch_size]))
    return torch.cat(feats, dim=0).mean(dim=0)                      # [D]


def _make_calib_fn(proxy, c_target, form, target, device):
    """Differentiable negative-calibration penalty of `delta` on an OOD batch."""
    if form == 'none':
        return None
    def fn(ood_batch, delta):
        x = torch.clamp(ood_batch + delta.unsqueeze(0), 0.0, 1.0)
        if form == 'cos':
            f = proxy.extract_feature(x)
            cos = F.cosine_similarity(f, c_target.unsqueeze(0), dim=1)
            return F.relu(cos).mean()
        if form == 'prob':
            return F.softmax(proxy(x), dim=1)[:, target].mean()
        raise ValueError(form)
    return fn


@torch.no_grad()
def _pool_asr(proxy, images, delta, target, device, eval_n=2000):
    if images is None or images.shape[0] == 0:
        return float('nan')
    return proxy_asr(proxy, images, delta, target, device, eval_n=eval_n)


# --------------------------------------------------------------------------- #
# A/B/C engine
# --------------------------------------------------------------------------- #
def optimize_ood_trigger(proxy, target_imgs, ood_imgs, nontarget_imgs, target, device,
                         pool='target', calib='none', calib_weight=1.0,
                         l2_budget=1.5, steps=8000, lr=0.02, batch_size=128,
                         seed=0, log_every=200, logger=print):
    """Optimise the universal trigger under an A/B/C arm. Returns (delta_np, info).

    pool : 'target' | 'target+ood' | 'nontarget'
    calib: 'none' | 'cos' | 'prob'      (requires ood_imgs; ignored when 'none')
    """
    set_random_seed(seed)
    H, W = target_imgs.shape[-2], target_imgs.shape[-1]

    if pool == 'target':
        pool_imgs = target_imgs
    elif pool == 'target+ood':
        assert ood_imgs is not None, 'pool=target+ood requires --ood_dataset'
        pool_imgs = torch.cat([target_imgs, ood_imgs], dim=0)
    elif pool == 'nontarget':
        pool_imgs = nontarget_imgs
    else:
        raise ValueError(pool)
    M = pool_imgs.shape[0]

    use_calib = calib != 'none'
    if use_calib:
        assert ood_imgs is not None, 'calib != none requires --ood_dataset'
        c_target = _feature_mean(proxy, target_imgs[:4000], device)
        calib_fn = _make_calib_fn(proxy, c_target, calib, target, device)
    else:
        c_target, calib_fn = None, None

    logger('[oodtrig] arm: pool=%s (%d imgs) calib=%s w=%.2f | OOD=%s | '
           'L2 budget=%.3f steps=%d lr=%.3f' %
           (pool, M, calib, calib_weight,
            'none' if ood_imgs is None else '%d imgs' % ood_imgs.shape[0],
            l2_budget, steps, lr))

    delta = torch.zeros(3, H, W, device=device).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    t0, history = time.time(), []
    for step in range(steps):
        bi = torch.randint(0, M, (min(batch_size, M),), device=device)
        x = pool_imgs[bi]
        logits = proxy(torch.clamp(x + delta.unsqueeze(0), 0.0, 1.0))
        tgt = torch.full((x.shape[0],), target, dtype=torch.long, device=device)
        L = F.cross_entropy(logits, tgt)
        L_cal = torch.zeros((), device=device)
        if use_calib:
            oi = torch.randint(0, ood_imgs.shape[0], (min(batch_size, ood_imgs.shape[0]),),
                               device=device)
            L_cal = calib_fn(ood_imgs[oi], delta)
            L = L + calib_weight * L_cal
        opt.zero_grad(); L.backward(); opt.step()

        with torch.no_grad():
            nrm = delta.norm()
            if nrm.item() > l2_budget:
                delta.data.mul_(l2_budget / (nrm + 1e-12))

        if step % log_every == 0 or step == steps - 1:
            d = delta.detach()
            rec = (step, float(L.item()), float(L_cal.item()), float(d.norm().item()),
                   _pool_asr(proxy, target_imgs, d, target, device),
                   _pool_asr(proxy, nontarget_imgs, d, target, device),
                   _pool_asr(proxy, ood_imgs, d, target, device),
                   time.time() - t0)
            history.append(rec)
            logger('[oodtrig %5d] L=%.4f cal=%.4f |d|=%.3f | ASR(tgt)=%.3f '
                   'ASR(nontgt)=%.3f ASR(ood)=%.3f (%.0fs)' % rec)

    d = delta.detach()
    info = {'pool': pool, 'calib': calib, 'calib_weight': calib_weight,
            'l2_budget': l2_budget, 'steps': steps, 'lr': lr,
            'pool_size': int(M),
            'ood_size': int(ood_imgs.shape[0]) if ood_imgs is not None else 0,
            'final_l2': float(d.norm().item()),
            'final_asr_target': _pool_asr(proxy, target_imgs, d, target, device),
            'final_asr_nontarget': _pool_asr(proxy, nontarget_imgs, d, target, device),
            'final_asr_ood': _pool_asr(proxy, ood_imgs, d, target, device),
            'history': history}
    logger('[oodtrig] DONE: |d|=%.3f ASR tgt/nontgt/ood = %.3f / %.3f / %.3f' %
           (info['final_l2'], info['final_asr_target'],
            info['final_asr_nontarget'], info['final_asr_ood']))
    return d.cpu().numpy(), info


# --------------------------------------------------------------------------- #
# CLI (standalone trigger-stage check -- no victim training)
# --------------------------------------------------------------------------- #
def _load_victim_dataset(args):
    if args.dataset == 'cifar10':
        return datasets.CIFAR10(root='./data', train=True,
                                transform=transforms.ToTensor(), download=True)
    if args.dataset == 'cifar100':
        return datasets.CIFAR100(root='./data100', train=True,
                                 transform=transforms.ToTensor(), download=True)
    tf = transforms.Compose([transforms.Resize(args.size), transforms.ToTensor()])
    return datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=tf)


def main():
    ap = argparse.ArgumentParser('OOD-calibrated trigger: A/B/C arms (48h mechanism check)')
    ap.add_argument('--mode', choices=['a', 'b', 'c', 'cur'], default='c',
                    help='a=positive-only | b=OOD-as-data | c=OOD-calibration | cur=frozen baseline')
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
    ap.add_argument('--ood_dataset', default='cifar100',
                    choices=['cifar10', 'cifar100', 'tiny', 'imagefolder'])
    ap.add_argument('--ood_data_dir', default='', help='for ood_dataset=imagefolder')
    ap.add_argument('--ood_cap', type=int, default=20000)
    ap.add_argument('--calib_weight', type=float, default=1.0)
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--save_dir', default='./resource_ood/triggers/trigger')
    args = ap.parse_args()

    arm = {'a': ('target', 'none'), 'b': ('target+ood', 'none'),
           'c': ('target', 'cos'), 'cur': ('nontarget', 'none')}[args.mode]
    pool, calib = arm
    if args.mode == 'c':
        calib = os.environ.get('OOD_CALIB_FORM', 'cos')   # allow 'prob' override
    set_random_seed(args.seed)
    proxy = load_proxy(args.proxy_path, args.num_classes, args.device)
    ds = _load_victim_dataset(args)
    imgs, lbls = build_image_and_label_tensors(ds, args.size, args.device)
    target_imgs, nontarget_imgs = imgs[lbls == args.y_target], imgs[lbls != args.y_target]
    ood_imgs = None
    if pool == 'target+ood' or calib != 'none':
        ood_imgs = load_ood_images(args.ood_dataset, args.ood_data_dir,
                                   args.size, args.device, cap=args.ood_cap)
    print('[oodtrig] mode=%s (pool=%s calib=%s) dataset=%s target=%d '
          '| tgt=%d nontgt=%d ood=%s' %
          (args.mode, pool, calib, args.dataset, args.y_target,
           target_imgs.shape[0], nontarget_imgs.shape[0],
           'n/a' if ood_imgs is None else ood_imgs.shape[0]))

    delta_np, info = optimize_ood_trigger(
        proxy, target_imgs, ood_imgs, nontarget_imgs, args.y_target, args.device,
        pool=pool, calib=calib, calib_weight=args.calib_weight,
        l2_budget=args.l2_budget, steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed, log_every=args.log_every)

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'global_delta.npy'), delta_np)
    meta = dict(vars(args)); meta['pool'], meta['calib'] = pool, calib
    meta.update({k: v for k, v in info.items() if k != 'history'})
    with open(os.path.join(args.save_dir, 'global_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('[oodtrig] saved -> %s' % args.save_dir)


if __name__ == '__main__':
    main()
