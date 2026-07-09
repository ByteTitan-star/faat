"""Ensemble-ICIT: train the input-conditioned generator on an ENSEMBLE of surrogate
architectures (R18+R34+R50) so the trigger is cross-architecture (transferable). Then
hand off to a victim (default ResNet18) for end-to-end ASR/BA.

Transfer claim is validated separately on HELD-OUT archs (ResNet101/152) -- NOT in the
ensemble -- via faat/_icit_blackbox_transfer.py.

  python -m faat.train_icit_ens --dataset cifar10 --num_classes 10 \
      --proxies resource/faat/proxy/resnet18_clean_cifar10.pth,resource/faat/proxy/resnet34_clean_cifar10.pth,resource/faat/proxy/resnet50_clean_cifar10.pth \
      --arches resnet18,resnet34,resnet50 --victim_arch resnet18 \
      --save_trigger resource/faat/icit_ens/cifar10_b2.0_s1 --budget 2.0 ...
"""
import argparse
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from cifar_resnet import ResNet18, ResNet34, ResNet50
from faat.ic_trigger import ICGenerator
from faat.global_trigger import proxy_argmax

ARCH = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50}
PY = sys.executable
DEV = 'cuda'


def load_proxy(path, arch, nc):
    ck = torch.load(path, map_location=DEV)
    m = ARCH[arch](num_classes=nc).to(DEV); m.load_state_dict(ck['state_dict']); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--proxies', required=True, help='comma-sep proxy paths')
    ap.add_argument('--arches', required=True, help='comma-sep arches matching proxies')
    ap.add_argument('--victim_arch', default='resnet18', help='victim architecture (for ASR/BA)')
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--poison_rate', type=float, default=0.01)
    ap.add_argument('--output_dir', default='./resource/save_metric_10_res')
    ap.add_argument('--select_epoch', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--save_trigger', required=True)
    ap.add_argument('--budget', type=float, default=2.0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--result_dir', required=True)
    ap.add_argument('--gpu', default=None)
    args = ap.parse_args()

    ppaths = args.proxies.split(','); arches = args.arches.split(',')
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True) if args.dataset == 'cifar10' \
        else datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transforms.ToTensor())
    imgs = torch.stack([ds[i][0] for i in range(min(len(ds), 12000))]).to(DEV)
    proxies = [load_proxy(pp, ar, args.num_classes) for pp, ar in zip(ppaths, arches)]
    # non-target pool (labels from first proxy)
    lbl = proxy_argmax(proxies[0], imgs, DEV)
    pool = torch.where(lbl != args.y_target)[0].to(DEV)
    imgs = imgs[pool]; M = len(imgs)
    tgt = torch.full((128,), args.y_target, dtype=torch.long, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(args.seed)

    gen = ICGenerator().to(DEV)
    opt = torch.optim.Adam(gen.parameters(), lr=5e-4)
    print('[icit-ens] training on ensemble %s, budget=%.2f, %d steps' % (arches, args.budget, args.steps))
    for step in range(args.steps):
        idx = torch.randint(0, M, (128,), generator=g); x = imgs[idx]
        xp = torch.clamp(x + gen(x, args.budget), 0, 1)
        loss = sum(F.cross_entropy(pr(xp), tgt) for pr in proxies) / len(proxies)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                fr = sum(pr(xp).argmax(1).eq(args.y_target).float().mean().item() for pr in proxies) / len(proxies)
            print('[icit-ens %5d] ce=%.4f ens_flipRate=%.3f' % (step + 1, loss.item(), fr))
    os.makedirs(args.save_trigger, exist_ok=True)
    torch.save({'state_dict': gen.state_dict(), 'budget': args.budget}, os.path.join(args.save_trigger, 'gen.pth'))
    print('[icit-ens] generator saved -> %s' % args.save_trigger)

    # hand off to victim training (icit branch works for any arch via --model)
    common = ("--dataset %s --model %s --epochs %d --learning_rate 0.1 --seed %d --y_target %d "
              "--poison_rate %g --output_dir %s --select_epoch %d --selection res --res_sel square "
              "--backdoor_type icit --icit_save_trigger %s --icit_budget %g --result_dir %s "
              "--num_classes %d --data_dir %s"
              ) % (args.dataset, args.victim_arch, args.epochs, args.seed, args.y_target, args.poison_rate,
                   args.output_dir, args.select_epoch, args.save_trigger, args.budget, args.result_dir,
                   args.num_classes, args.data_dir)
    cmd = "%s -u train_backdoor.py %s" % (PY, common)
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('[icit-ens] launching victim (%s): %s' % (args.victim_arch, cmd))
    subprocess.run(cmd, shell=True, env=env, check=False)


if __name__ == '__main__':
    main()
