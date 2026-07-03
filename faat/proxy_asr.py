"""Cheap ASR proxy (no victim training): how well does a global trigger direction
fool the FROZEN CLEAN proxy?

proxy_asr = fraction of non-target test images the clean proxy classifies as `target`
            after clamp(x + delta_global).

This is exactly the Narcissus objective evaluated on the frozen proxy. Victim-ASR
(trained on poisoned data) is usually >= proxy_asr (poisoning amplifies the trigger),
so proxy_asr is a conservative, fast (~seconds) filter to map the (L2, ASR) Pareto
frontier before spending 3h per victim training.

Run:
  python -m faat.proxy_asr --source narcissus --scales 1.0,0.5,0.3,0.15,0.1 --target 0
  python -m faat.proxy_asr --source artifact --artifact resource/faat/save_trigger_10_0/global_delta.npy --scales 1.0,0.5,0.3 --target 0
"""
import os
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from .proxy import load_proxy

NAR_NOISE = './resource/narcissus/noise_01000.pth'


def _base_delta(source, artifact):
    if source == 'narcissus':
        return torch.load(NAR_NOISE, map_location='cpu').squeeze(0).float()
    return torch.from_numpy(np.load(artifact)).float()


@torch.no_grad()
def proxy_asr(delta_global, proxy, target, device, n=2000):
    ds = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
    idx = [i for i in range(len(ds)) if int(ds[i][1]) != target][:n]
    x = torch.stack([ds[i][0] for i in idx]).to(device)
    out = proxy(torch.clamp(x + delta_global.to(device), 0.0, 1.0))
    return float((out.argmax(1) == target).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['narcissus', 'artifact'], default='narcissus')
    ap.add_argument('--artifact', default='./resource/faat/save_trigger_10_0/global_delta.npy')
    ap.add_argument('--scales', default='1.0,0.5,0.3,0.15,0.1')
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--proxy_path',
                    default='./resource/faat/proxy/resnet18_clean_cifar10.pth')
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n', type=int, default=2000)
    args = ap.parse_args()

    proxy = load_proxy(args.proxy_path, args.num_classes, args.device)
    base = _base_delta(args.source, args.artifact)
    base_l2 = float(base.norm().item())
    print('source=%s  base L2=%.3f  target=%d  n=%d' % (args.source, base_l2, args.target, args.n))
    for sc in [float(s) for s in args.scales.split(',')]:
        d = base * sc
        asr = proxy_asr(d, proxy, args.target, args.device, args.n)
        print('  scale=%.3f  L2=%6.3f  proxy_ASR=%.4f' % (sc, float(d.norm().item()), asr))


if __name__ == '__main__':
    main()
