"""FAAT Stage B offline optimisation.

Freeze the clean proxy + c_target; optimise {delta_global, pi_theta, UnetGenerator}
to minimise a weighted combination of:
    L_align       : ||f(x') - c_target||_2            (core C-mechanism)
    L_align_global: ||f(clamp(x+delta_global)) - c_target||_2
                                                 (C2 health: test-time uses delta_global only)
    L_perc (=1-SSIM), L_l2, L_freq(high-band)        (stealth / augmentation-robustness)

Artifacts written to save_trigger/:
    global_delta.npy [3,H,W]   adaptive_delta.npy [N,3,H,W]   poison_keys.npy [N]
    c_target.npy [D]   strategy.pth   unet.pth   opt.log   meta.json

`run_optimization(cfg)` is the callable; train_faat.py and the smoke test both use it.
Run standalone:  python -m faat.optimize --dataset cifar10 --y_target 0 ...
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
from .proxy import (select_poison_inds, load_proxy, train_clean_proxy,
                    compute_center, build_summaries, stack_poison_images)
from .strategy_net import StrategyNet, N_BANDS
from .trigger_gen import FAATGenerator
from .losses import align_loss, ssim_loss, l2_loss, freq_loss

NAR_NOISE = './resource/narcissus/noise_01000.pth'


def _init_global_direction(size, scale, init_random, device):
    """Warm-start delta_global from the validated Narcissus direction (or random)."""
    if not init_random and os.path.exists(NAR_NOISE):
        t = torch.load(NAR_NOISE, map_location='cpu').squeeze(0).float()
        return (t * float(scale)).to(device)
    g = torch.Generator().manual_seed(12345)
    return (0.02 * torch.randn((3, size, size), generator=g)).to(device)


def run_optimization(cfg):
    device = cfg.device
    set_random_seed(cfg.seed)

    # 1) data + identical poison-index selection as the host pipeline
    if cfg.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                         transform=transforms.ToTensor(), download=True)
    elif cfg.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data100', train=True,
                                          transform=transforms.ToTensor(), download=True)
    else:
        train_dataset = datasets.ImageFolder(root=os.path.join(cfg.data_dir, 'train'),
                                             transform=transforms.ToTensor())
    poison_inds, _ = select_poison_inds(
        train_dataset, cfg.selection, cfg.output_dir, cfg.select_epoch, cfg.seed,
        cfg.num_classes, cfg.y_target, cfg.res_sel, cfg.poison_rate, cfg.res_rate)
    if cfg.smoke_n is not None and cfg.smoke_n < len(poison_inds):
        poison_inds = poison_inds[:cfg.smoke_n]
    print('[faat-opt] %d poison candidates (selection=%s res_sel=%s y_target=%d)' %
          (len(poison_inds), cfg.selection, cfg.res_sel, cfg.y_target))

    # 2) proxy (train if missing) + c_target + cached summaries
    if cfg.proxy_path and not os.path.exists(cfg.proxy_path) and cfg.train_proxy_if_missing:
        print('[faat-opt] proxy missing, training clean proxy (%d ep)...' % cfg.proxy_epochs)
        train_clean_proxy(dataset=cfg.dataset, num_classes=cfg.num_classes,
                          epochs=cfg.proxy_epochs, device=device,
                          save_path=cfg.proxy_path, seed=cfg.seed,
                          data_dir=getattr(cfg, 'data_dir', './data'))
    proxy = load_proxy(cfg.proxy_path, cfg.num_classes, device)
    if not (cfg.proxy_path and os.path.exists(cfg.proxy_path)):
        print('[faat-opt] WARNING: no proxy checkpoint -> using RANDOM init '
              '(only OK for smoke test; real runs must train a proxy).')
    c_target = compute_center(proxy, train_dataset, cfg.y_target, device).to(device)
    x_poison = stack_poison_images(train_dataset, poison_inds)              # [N,3,H,W]
    S = build_summaries(proxy, x_poison, c_target, device, size=cfg.size,
                        n_bands=cfg.n_bands)                                # [N,d_s]

    # 3) generator + policy
    init_g = _init_global_direction(cfg.size, cfg.init_global_scale,
                                    cfg.init_random, device)
    gen = FAATGenerator(size=cfg.size, n_bands=cfg.n_bands, eps_max=cfg.eps_max,
                        sparse_gate=cfg.sparse_gate, init_global=init_g,
                        adaptive_l2_max=getattr(cfg, 'adaptive_l2_max', 0.0)).to(device)
    policy = StrategyNet(n_bands=cfg.n_bands).to(device)
    band_idx = gen.band_idx                                                # [H,W] on device

    # v3: optionally freeze delta_global (keep the strong Narcissus direction as-is;
    # optimise only the adaptive residual + policy for defense-evasion shaping).
    if getattr(cfg, 'fix_global', False):
        gen.delta_global.requires_grad_(False)
        log_msg = 'fix_global=True (delta_global frozen at Narcissus x %.3f)' % cfg.init_global_scale
        print('[faat-opt] ' + log_msg)

    # 4) optimisers (delta_global gets its own lr per plan 4.6)
    params_adaptive = list(gen.unet.parameters()) + list(policy.parameters())
    groups = [{'params': params_adaptive, 'lr': cfg.lr_policy}]
    if not getattr(cfg, 'fix_global', False):
        groups.append({'params': [gen.delta_global], 'lr': cfg.lr_global})
    opt = torch.optim.Adam(groups)

    os.makedirs(cfg.save_trigger, exist_ok=True)
    log_path = os.path.join(cfg.save_trigger, 'opt.log')
    logf = open(log_path, 'w')

    def log(msg):
        print(msg); logf.write(msg + '\n'); logf.flush()

    N = len(poison_inds)
    x_poison = x_poison.to(device)
    S = S.to(device)
    t0 = time.time()
    log('[faat-opt] steps=%d bs=%d eps_max=%.3f lambdas(a=%.2f ag=%.2f p=%.2f '
        'l2=%.2f f=%.3f)' % (cfg.steps, cfg.batch_size, cfg.eps_max,
        cfg.lambda_align, cfg.lambda_align_global, cfg.lambda_perc,
        cfg.lambda_l2, cfg.lambda_freq))

    # 5) optimisation loop
    for step in range(cfg.steps):
        idx = torch.randint(0, N, (min(cfg.batch_size, N),), device=device)
        x_b = x_poison[idx]
        s_b = S[idx]
        pol = policy(s_b)
        x_prime, da, dg = gen(x_b, pol)

        feat = proxy.extract_feature(x_prime)
        L_align = align_loss(feat, c_target)
        L = cfg.lambda_align * L_align

        L_align_g = torch.zeros((), device=device)
        if not getattr(cfg, 'fix_global', False):
            x_g = torch.clamp(x_b + dg, 0.0, 1.0)
            if cfg.global_obj == 'asr':
                # v2: delta_global optimised to FOOL the proxy into predicting target
                # (the Narcissus objective -- a direct ASR surrogate, unlike L_align).
                tgt = torch.full((x_b.shape[0],), cfg.y_target, dtype=torch.long, device=device)
                L_align_g = F.cross_entropy(proxy(x_g), tgt)
                L = L + cfg.lambda_asr_global * L_align_g
            elif cfg.lambda_align_global > 0:
                L_align_g = align_loss(proxy.extract_feature(x_g), c_target)
                L = L + cfg.lambda_align_global * L_align_g

        if cfg.lambda_perc > 0:
            L = L + cfg.lambda_perc * ssim_loss(x_b, x_prime)
        if cfg.lambda_l2 > 0:
            L = L + cfg.lambda_l2 * (l2_loss(da) + l2_loss(dg))
        if cfg.lambda_freq > 0:
            L = L + cfg.lambda_freq * (freq_loss(da, band_idx, cfg.n_bands)
                                       + freq_loss(dg, band_idx, cfg.n_bands))

        opt.zero_grad()
        L.backward()
        opt.step()

        # hard L2 budget on delta_global (precise stealth control for the Pareto sweep;
        # 0 = unconstrained). Projective: rescale onto the L2 ball when exceeded.
        if cfg.global_l2_max and cfg.global_l2_max > 0:
            with torch.no_grad():
                nrm = gen.delta_global.norm()
                if nrm.item() > cfg.global_l2_max:
                    gen.delta_global.mul_(cfg.global_l2_max / (nrm + 1e-12))

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                da_l2 = l2_loss(da).item()
                dg_l2 = l2_loss(dg).item()
            log('[step %4d] L=%.4f align=%.4f align_g=%.4f '
                '|da|=%.3f |dg|=%.3f  (%.1fs)' %
                (step, L.item(), L_align.item(), L_align_g.item(),
                 da_l2, dg_l2, time.time() - t0))

    # 6) final pass: materialise per-sample adaptive residuals
    gen.eval(); policy.eval()
    adaptive = np.zeros((N, 3, cfg.size, cfg.size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, cfg.batch_size):
            sl = slice(s, min(s + cfg.batch_size, N))
            pol = policy(S[sl])
            _, da, _ = gen(x_poison[sl], pol)
            adaptive[s:sl.stop] = da.cpu().numpy()

    # 7) save artifacts
    np.save(os.path.join(cfg.save_trigger, 'global_delta.npy'),
            gen.delta_global.detach().cpu().numpy())
    np.save(os.path.join(cfg.save_trigger, 'adaptive_delta.npy'), adaptive)
    np.save(os.path.join(cfg.save_trigger, 'poison_keys.npy'),
            np.asarray(poison_inds, dtype=np.int64))
    np.save(os.path.join(cfg.save_trigger, 'c_target.npy'),
            c_target.detach().cpu().numpy())
    torch.save(policy.state_dict(), os.path.join(cfg.save_trigger, 'strategy.pth'))
    torch.save(gen.unet.state_dict(), os.path.join(cfg.save_trigger, 'unet.pth'))
    meta = {
        'y_target': cfg.y_target, 'selection': cfg.selection, 'res_sel': cfg.res_sel,
        'poison_rate': cfg.poison_rate, 'seed': cfg.seed,
        'n_poison': int(N), 'steps': cfg.steps, 'eps_max': cfg.eps_max,
        'lambda_align': cfg.lambda_align, 'lambda_align_global': cfg.lambda_align_global,
        'lambda_perc': cfg.lambda_perc, 'lambda_l2': cfg.lambda_l2,
        'lambda_freq': cfg.lambda_freq, 'init_random': cfg.init_random,
        'init_global_scale': cfg.init_global_scale,
        'global_obj': cfg.global_obj, 'global_l2_max': cfg.global_l2_max,
        'lambda_asr_global': cfg.lambda_asr_global,
        'final_align': float(L_align.item()), 'final_align_global': float(L_align_g.item()),
    }
    with open(os.path.join(cfg.save_trigger, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('[faat-opt] DONE -> %s  (final align=%.4f align_g=%.4f, %.1fs)' %
        (cfg.save_trigger, L_align.item(), L_align_g.item(), time.time() - t0))
    logf.close()
    return cfg.save_trigger, meta


def build_argparser():
    p = argparse.ArgumentParser('FAAT Stage B offline optimisation')
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--data_dir', default='./data',
                   help='ImageFolder root for non-cifar datasets (expects train/ + val/ subdirs)')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--selection', default='res')
    p.add_argument('--res_sel', default='square')
    p.add_argument('--output_dir', default='./resource/save_metric_10_res')
    p.add_argument('--select_epoch', type=int, default=10)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--y_target', type=int, default=0)
    p.add_argument('--poison_rate', type=float, default=0.01)
    p.add_argument('--res_rate', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--proxy_path', default='./resource/faat/proxy/resnet18_clean_cifar10.pth')
    p.add_argument('--proxy_epochs', type=int, default=100)
    p.add_argument('--train_proxy_if_missing', action='store_true')
    p.add_argument('--save_trigger', default='./resource/faat/save_trigger_10_0')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=48)
    p.add_argument('--lr_policy', type=float, default=1e-3)
    p.add_argument('--lr_global', type=float, default=5e-4)
    p.add_argument('--global_l2_max', type=float, default=0.0,
                   help='hard L2 budget on delta_global (0=unconstrained); for Pareto stealth sweep')
    p.add_argument('--global_obj', choices=['align', 'asr'], default='align',
                   help="delta_global objective: 'align'=L_align(feature, v1) | 'asr'=CE fool proxy (Narcissus-style, v2)")
    p.add_argument('--lambda_asr_global', type=float, default=1.0,
                   help='weight of the ASR-proxy CE loss on delta_global (global_obj=asr)')
    p.add_argument('--fix_global', action='store_true',
                   help='v3: freeze delta_global at the Narcissus init (do NOT optimise it); '
                        'only optimise delta_adaptive + policy for defense-evasion shaping')
    p.add_argument('--adaptive_l2_max', type=float, default=0.0,
                   help='v3.1: hard per-sample L2 budget on delta_adaptive (0=unbounded). '
                        'Keep << |delta_global| so adaptive cannot become a train-only co-trigger (C2).')
    p.add_argument('--eps_max', type=float, default=0.05)
    p.add_argument('--n_bands', type=int, default=N_BANDS)
    p.add_argument('--size', type=int, default=32)
    p.add_argument('--sparse_gate', action='store_true')
    p.add_argument('--init_global_scale', type=float, default=1.0)
    p.add_argument('--init_random', action='store_true')
    p.add_argument('--lambda_align', type=float, default=1.0)
    p.add_argument('--lambda_align_global', type=float, default=0.5)
    p.add_argument('--lambda_perc', type=float, default=0.3)
    p.add_argument('--lambda_l2', type=float, default=0.05)
    p.add_argument('--lambda_freq', type=float, default=0.02)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--smoke_n', type=int, default=None)
    return p


if __name__ == '__main__':
    cfg = build_argparser().parse_args()
    run_optimization(cfg)
