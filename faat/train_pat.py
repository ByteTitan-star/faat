"""PAT runner: optimize the perceptually-allocated universal delta on a clean proxy, save,
hand off to train_backdoor.py --backdoor_type pat for victim training.

Run: CUDA_VISIBLE_DEVICES=0 python -u -m faat.train_pat --alpha 1.0 --y_target 0 --seed 1 \
  --result_dir results/pat_a1.0_s1 --save_trigger resource/faat/pat/a1.0
"""
import argparse
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.pat_trigger import optimize_pat_trigger, save_pat_trigger

DEV = 'cuda'


def load_proxy(path, nc=10):
    ck = torch.load(path, map_location=DEV)
    m = ResNet18(num_classes=nc).to(DEV); m.load_state_dict(ck['state_dict']); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--proxy', default='resource/faat/proxy/resnet18_clean_cifar10.pth')
    ap.add_argument('--save_trigger', default='resource/faat/pat/a1.0')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--train_epochs', type=int, default=300)
    ap.add_argument('--result_dir', default='results/pat_a1.0_s1')
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)
    proxy = load_proxy(args.proxy, args.num_classes)
    info = optimize_pat_trigger(proxy, imgs, args.y_target, DEV, alpha=args.alpha,
                                steps=args.steps, seed=args.seed)
    save_pat_trigger(info['delta'], args.save_trigger, args.alpha, info['final_proxy_asr'])
    print('[pat] saved trigger to %s (proxyASR=%.4f)' % (args.save_trigger, info['final_proxy_asr']))

    if args.skip_train:
        return
    cmd = [sys.executable, '-u', '-m', 'train_backdoor',
           '--dataset', 'cifar10', '--model', 'resnet18', '--backdoor_type', 'pat',
           '--num_classes', str(args.num_classes), '--y_target', str(args.y_target),
           '--poison_rate', str(args.poison_rate), '--seed', str(args.seed),
           '--epochs', str(args.train_epochs), '--pat_save_trigger', args.save_trigger,
           '--result_dir', args.result_dir, '--output_dir', './resource/save_metric_10_res',
           '--select_epoch', '10', '--selection', 'res', '--res_sel', 'square']
    print('[pat] launching victim: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
