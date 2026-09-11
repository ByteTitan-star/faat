"""OPAL runner: generate the secret ordinal key, save it, smoke-check projection (fraction
satisfied + stealth), then hand off to train_backdoor.py --backdoor_type opal.

OPAL has NO proxy objective -- its learnability is a pure empirical question. We sweep block size
and margin (the two knobs that trade stealth vs learnability).

Run: CUDA_VISIBLE_DEVICES=0 python -u -m faat.train_opal --block 4 --M 128 --margin 0.016 \
    --y_target 0 --seed 1 --result_dir results/opal_b4_m0.016_s1 \
    --save_trigger resource/faat/opal/b4_m0.016_s1
"""
import argparse
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from faat.opal_trigger import make_key, save_opal_trigger, opal_project, opal_score
# make_key returns (U,V,C,level); we keep key=(U,V,C) for projection/save

DEV = 'cuda'


def _stealth_check(key, margin, block, linf, device='cuda'):
    """Quick projection smoke: fraction satisfied, L_inf, mean |delta| on a few CIFAR images."""
    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    x = torch.stack([ds[i][0] for i in range(256)]).to(device)
    z, delta = opal_project(x, key, margin, block=block, linf=linf)
    sat = opal_score(z, key, margin, block=block)
    linf_emp = (z - x).abs().max().item()
    l2 = (z - x).flatten(1).norm(dim=1).mean().item()
    print('[opal-smoke] satisfied=%.3f L_inf=%.5f meanL2=%.4f (margin=%.5f block=%d linf=%s)'
          % (sat, linf_emp, l2, margin, block, linf))
    return sat, linf_emp, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--save_trigger', default='resource/faat/opal/b4_m0.016_s1')
    ap.add_argument('--block', type=int, default=4, help='block size in px (32/block = grid side)')
    ap.add_argument('--M', type=int, default=128, help='number of secret ordinal pairs')
    ap.add_argument('--margin', type=float, default=4.0 / 255, help='ordinal margin (0-1 scale)')
    ap.add_argument('--levels', default='binary', choices=['binary', 'cont'],
                    help='latent level assignment: binary (2-group, stealthy) or cont (full ranking)')
    ap.add_argument('--linf', type=float, default=8.0 / 255,
                    help='per-pixel L_inf budget (clamp delta -> stealthy; None=unbounded full projection)')
    ap.add_argument('--key_seed', type=int, default=1)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--train_epochs', type=int, default=300)
    ap.add_argument('--result_dir', default='results/opal_b4_m0.016_s1')
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    n_blocks = (32 // args.block) ** 2
    U, V, C, level = make_key(n_blocks, args.M, seed=args.key_seed, levels=args.levels)
    key = (U, V, C)
    linf = args.linf if args.linf > 0 else None
    save_opal_trigger(key, args.margin, args.block, args.save_trigger, linf=linf)
    print('[opal] saved key to %s (block=%d M=%d n_blocks=%d margin=%.5f levels=%s linf=%s n_high=%d)'
          % (args.save_trigger, args.block, args.M, n_blocks, args.margin, args.levels, linf, int((level > 0).sum())))
    _stealth_check(key, args.margin, args.block, linf, device=DEV)

    if args.skip_train:
        return
    cmd = [sys.executable, '-u', '-m', 'train_backdoor',
           '--dataset', 'cifar10', '--model', 'resnet18', '--backdoor_type', 'opal',
           '--num_classes', str(args.num_classes), '--y_target', str(args.y_target),
           '--poison_rate', str(args.poison_rate), '--seed', str(args.seed),
           '--epochs', str(args.train_epochs), '--opal_save_trigger', args.save_trigger,
           '--result_dir', args.result_dir, '--output_dir', './resource/save_metric_10_res',
           '--select_epoch', '10', '--selection', 'res', '--res_sel', 'square']
    print('[opal] launching victim: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
