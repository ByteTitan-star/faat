"""FEAST runner: optimize the universal phase trigger on a clean proxy, save, then hand off to
train_backdoor.py --backdoor_type feast for victim training (starvation happens at injection).

Run (4-GPU validation):
  CUDA_VISIBLE_DEVICES=0 python -u -m faat.train_feast --starve_eps 0.031 --y_target 0 --seed 1 \
      --result_dir results/feast_full_s1 --save_trigger resource/faat/feast/full_s1
  # ablation (phase only, no starvation):
  CUDA_VISIBLE_DEVICES=2 python -u -m faat.train_feast --starve_eps 0 --y_target 0 --seed 1 \
      --result_dir results/feast_phaseonly_s1 --save_trigger resource/faat/feast/phaseonly_s1
"""
import argparse
import subprocess
import sys

import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.feast_trigger import optimize_feast_phase, optimize_feast_pixel, save_feast_trigger

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
    ap.add_argument('--save_trigger', default='resource/faat/feast/full_s1')
    ap.add_argument('--trigger_mode', default='pixel', choices=['pixel', 'phase'],
                    help="pixel=Narcissus-style effective trigger (isolates starvation); phase=user's FFT-phase design")
    ap.add_argument('--phi_max', type=float, default=0.6, help='phase magnitude bound (rad)')
    ap.add_argument('--radius', type=int, default=8, help='low-frequency radius (bins)')
    ap.add_argument('--budget', type=float, default=1.0, help='pixel mode: universal-delta L2 budget')
    ap.add_argument('--starve_eps', type=float, default=8.0/255, help='starvation L_inf budget; 0=no starvation')
    ap.add_argument('--starve_steps', type=int, default=30)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--train_epochs', type=int, default=300)
    ap.add_argument('--result_dir', default='results/feast_full_s1')
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)
    proxy = load_proxy(args.proxy, args.num_classes)
    if args.trigger_mode == 'phase':
        info = optimize_feast_phase(proxy, imgs, args.y_target, DEV,
                                    phi_max=args.phi_max, radius=args.radius, steps=args.steps,
                                    seed=args.seed)
    else:
        info = optimize_feast_pixel(proxy, imgs, args.y_target, DEV,
                                    budget=args.budget, steps=args.steps, seed=args.seed)
    save_feast_trigger(info, args.save_trigger)
    print('[feast] saved %s trigger to %s (proxyASR=%.4f, starve_eps=%.4f)'
          % (info['mode'], args.save_trigger, info['final_proxy_asr'], args.starve_eps))

    if args.skip_train:
        return
    cmd = [sys.executable, '-u', '-m', 'train_backdoor',
           '--dataset', 'cifar10', '--model', 'resnet18', '--backdoor_type', 'feast',
           '--num_classes', str(args.num_classes), '--y_target', str(args.y_target),
           '--poison_rate', str(args.poison_rate), '--seed', str(args.seed),
           '--epochs', str(args.train_epochs), '--feast_save_trigger', args.save_trigger,
           '--feast_starve_eps', str(args.starve_eps), '--feast_starve_steps', str(args.starve_steps),
           '--result_dir', args.result_dir, '--output_dir', './resource/save_metric_10_res',
           '--select_epoch', '10', '--selection', 'res', '--res_sel', 'square']
    print('[feast] launching victim: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
