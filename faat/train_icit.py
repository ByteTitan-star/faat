"""ICIT runner: train the input-conditioned generator (offline, frozen proxy), then hand
off to train_backdoor.py --backdoor_type icit for victim training.

  python -m faat.train_icit --dataset cifar10 --num_classes 10 \
      --proxy_path resource/faat/proxy/resnet18_clean_cifar10.pth \
      --save_trigger resource/faat/icit/cifar10_l2_1.5_seed1 \
      --budget 1.5 --poison_rate 0.01 --y_target 0 --seed 1 --epochs 300 \
      --result_dir results/icit_cifar10_l2_1.5_seed1
"""
import argparse
import os
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from .ic_trigger import optimize_ic_trigger, save_ic_trigger
from .proxy import load_proxy

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--proxy_path', required=True)
    ap.add_argument('--train_proxy_if_missing', action='store_true')
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--output_dir', default='./resource/save_metric_10_res')
    ap.add_argument('--select_epoch', type=int, default=10)
    ap.add_argument('--selection', default='res')
    ap.add_argument('--res_sel', default='square')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--save_trigger', required=True)
    ap.add_argument('--budget', type=float, default=1.5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--result_dir', required=True)
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--train', action='store_true', default=True)
    args = ap.parse_args()
    device = 'cuda'

    # 1) load clean images (non-cifar via ImageFolder)
    if args.dataset == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    elif args.dataset == 'cifar100':
        ds = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=True)
    else:
        ds = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transforms.ToTensor())
    imgs = torch.stack([ds[i][0] for i in range(min(len(ds), 12000))]).to(device)

    # 2) train generator (proxy if missing)
    if args.proxy_path and not os.path.exists(args.proxy_path) and args.train_proxy_if_missing:
        from .proxy import train_clean_proxy
        train_clean_proxy(dataset=args.dataset, num_classes=args.num_classes, data_dir=args.data_dir,
                          epochs=100, device=device, save_path=args.proxy_path, seed=args.seed)
    proxy = load_proxy(args.proxy_path, args.num_classes, device)
    res = optimize_ic_trigger(proxy, imgs, args.y_target, device, budget=args.budget,
                              steps=args.steps, seed=args.seed)
    save_ic_trigger(res['gen'], args.save_trigger, args.budget, res['final_proxy_asr'])
    print('[train_icit] generator saved -> %s (proxyASR=%.4f)' % (args.save_trigger, res['final_proxy_asr']))

    # 3) hand off to victim training
    common = (
        "--dataset %s --model resnet18 --epochs %d --learning_rate 0.1 --seed %d "
        "--y_target %d --poison_rate %g --output_dir %s --select_epoch %d "
        "--selection %s --res_sel %s --backdoor_type icit "
        "--icit_save_trigger %s --icit_budget %g --result_dir %s "
        "--num_classes %d --data_dir %s"
    ) % (args.dataset, args.epochs, args.seed, args.y_target, args.poison_rate,
         args.output_dir, args.select_epoch, args.selection, args.res_sel,
         args.save_trigger, args.budget, args.result_dir, args.num_classes, args.data_dir)
    cmd = "%s -u train_backdoor.py %s" % (PY, common)
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('[train_icit] launching victim training: %s' % cmd)
    subprocess.run(cmd, shell=True, env=env, check=False)


if __name__ == '__main__':
    main()
