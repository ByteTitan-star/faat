"""ORBIT-IRREP runner: build the D4-orbit base patch b (hand-designed asymmetric, instant), save,
then hand off to train_backdoor.py --backdoor_type orbit for victim training.

The host pipeline applies Pad+RandomHorizontalFlip+RandomCrop AFTER injection, so flip maps the
placed orbit element to another orbit element (same irrep signature) -> augmentation-closed.
Optional --opt_steps refines b with group-augmented proxy CE (weak for small patches; off by
default since patch triggers are learned directly by the victim, like Badnets).

Run (CIFAR-10, 1% poison, default k=5 at crop-safe off-center loc):
  CUDA_VISIBLE_DEVICES=0 python -u -m faat.train_orbit --y_target 0 --seed 1 \
    --result_dir results/orbit_k5_s1 --save_trigger resource/faat/orbit/k5
"""
import argparse
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.orbit_trigger import build_orbit_trigger, save_orbit_trigger, verify_invariance

DEV = 'cuda'


def load_proxy(path, nc=10):
    ck = torch.load(path, map_location=DEV)
    m = ResNet18(num_classes=nc).to(DEV); m.load_state_dict(ck['state_dict']); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _loc(s):
    a, b = s.split(',')
    return (int(a), int(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--proxy', default='resource/faat/proxy/resnet18_clean_cifar10.pth')
    ap.add_argument('--save_trigger', default='resource/faat/orbit/k5')
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--k', type=int, default=5, help='base patch size k x k (5 or 7)')
    ap.add_argument('--loc', type=_loc, default=(21, 21), help='patch top-left (r0,c0); crop-safe')
    ap.add_argument('--channels', type=int, default=3, help='1=grayscale-structural broadcast, 3=color')
    ap.add_argument('--contrast', type=float, default=1.0, help='patch contrast (1.0 = hard 0/1 overwrite)')
    ap.add_argument('--opt_steps', type=int, default=0, help='0=hand-designed base (default); >0=proxy-refine b')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--train_epochs', type=int, default=300)
    ap.add_argument('--result_dir', default='results/orbit_k5_s1')
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--strong_aug', action='store_true', help='add ColorJitter+RandomErasing (stronger aug-closure test)')
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    proxy = images = None
    if args.opt_steps:
        proxy = load_proxy(args.proxy, args.num_classes)
        ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
        images = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)

    info = build_orbit_trigger(k=args.k, channels=args.channels, loc=tuple(args.loc), seed=args.seed,
                               contrast=args.contrast, proxy=proxy, images=images, target=args.y_target,
                               device=DEV, opt_steps=args.opt_steps)
    sig = verify_invariance(info['base'].cpu())
    print('[orbit] D4-invariant signature (verified identical for all 8 orbit elements):')
    for name in ['A1', 'A2', 'B1', 'B2', 'E']:
        print('         E_%s = %.6e' % (name, sig[name]))
    save_orbit_trigger(info, args.save_trigger)
    print('[orbit] saved trigger to %s' % args.save_trigger)

    if args.skip_train:
        return
    cmd = [sys.executable, '-u', '-m', 'train_backdoor',
           '--dataset', 'cifar10', '--model', 'resnet18', '--backdoor_type', 'orbit',
           '--num_classes', str(args.num_classes), '--y_target', str(args.y_target),
           '--poison_rate', str(args.poison_rate), '--seed', str(args.seed),
           '--epochs', str(args.train_epochs), '--orbit_save_trigger', args.save_trigger,
           '--result_dir', args.result_dir, '--output_dir', './resource/save_metric_10_res',
           '--select_epoch', '10', '--selection', 'res', '--res_sel', 'square']
    if args.strong_aug:
        cmd.append('--strong_aug')
    print('[orbit] launching victim: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
