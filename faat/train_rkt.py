"""RKT probe runner: optimize the resampling kernel on a clean proxy, save it, then hand
off to train_backdoor.py --backdoor_type rkt for victim training.

Run: CUDA_VISIBLE_DEVICES=0 python -u -m faat.train_rkt --scale 0.7 --ksize 5 \
       --y_target 0 --seed 1 --result_dir results/rkt_cifar10_s0.7_s1
"""
import argparse
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.rkt_trigger import optimize_rkt_trigger, save_rkt_trigger

DEV = 'cuda'


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
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--proxy', default='resource/faat/proxy/resnet18_clean_cifar10.pth')
    ap.add_argument('--save_trigger', default='resource/faat/rkt/cifar10_s0.7_k5_s1')
    ap.add_argument('--scale', type=float, default=0.7)
    ap.add_argument('--ksize', type=int, default=5)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--stealth_lam', type=float, default=1e-2)
    ap.add_argument('--train_epochs', type=int, default=300)
    ap.add_argument('--result_dir', default='results/rkt_cifar10_s0.7_s1')
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)

    proxy = load_proxy(args.proxy, args.num_classes)
    info = optimize_rkt_trigger(proxy, imgs, args.y_target, DEV, scale=args.scale,
                                ksize=args.ksize, steps=args.steps, stealth_lam=args.stealth_lam,
                                seed=args.seed)
    save_rkt_trigger(info['trig'], args.save_trigger, args.scale, args.ksize, info['final_proxy_asr'])
    print('[rkt] saved trigger to %s (proxyASR=%.4f)' % (args.save_trigger, info['final_proxy_asr']))

    if args.skip_train:
        return
    cmd = [sys.executable, '-u', '-m', 'train_backdoor',
           '--dataset', args.dataset, '--model', 'resnet18', '--backdoor_type', 'rkt',
           '--num_classes', str(args.num_classes), '--y_target', str(args.y_target),
           '--poison_rate', str(args.poison_rate), '--seed', str(args.seed),
           '--epochs', str(args.train_epochs), '--rkt_save_trigger', args.save_trigger,
           '--result_dir', args.result_dir, '--output_dir', './resource/save_metric_10_res',
           '--select_epoch', '10', '--selection', 'res', '--res_sel', 'square']
    print('[rkt] launching victim training: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
