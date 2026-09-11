"""Path 3a: cross-architecture transfer test (the decisive path-3 experiment).

Trigger optimized on ResNet18 (source). Does it flip ResNet34/ResNet50 (target)?
  * HIGH transfer (>50%) -> Narcissus/ICIT are robust across architectures (no headroom).
  * LOW transfer (<30%)  -> cross-arch transfer is a weakness -> NOVEL headroom: a
    'transferable invisible trigger' becomes a real contribution (path 3 viable).

Tests both Narcissus (universal delta) and ICIT (input-conditioned generator).
"""
import numpy as np
import torch
from torchvision import datasets, transforms
from cifar_resnet import ResNet18, ResNet34, ResNet50
from faat.ic_trigger import load_ic_generator

DEV = 'cuda'
ARCH = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50}


def load_proxy(path, arch, nc, dev):
    ck = torch.load(path, map_location=dev)
    m = ARCH[arch](num_classes=nc).to(dev)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


@torch.no_grad()
def flip_rate(model, imgs, target, apply):
    """Fraction of clean non-target images the model classifies as target after apply(x)."""
    out = model(apply(imgs))
    return out.argmax(1).eq(target).float().mean().item()


def main():
    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(3000)]).to(DEV)
    target = 0

    # source triggers
    nar_delta = torch.from_numpy(np.load('resource/faat/save_trigger_10_0/global_delta.npy')).to(DEV)
    nar_apply = lambda x: torch.clamp(x + nar_delta, 0, 1)
    icit_gen, icit_b = load_ic_generator('resource/faat/icit/cifar10_l2_2.0_seed1', DEV)
    icit_apply = lambda x: torch.clamp(x + icit_gen(x, icit_b), 0, 1)
    clean_apply = lambda x: x

    proxies = {
        'resnet18': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
        'resnet34': 'resource/faat/proxy/resnet34_clean_cifar10.pth',
        'resnet50': 'resource/faat/proxy/resnet50_clean_cifar10.pth',
    }
    print('%-10s %10s %12s %12s' % ('arch', 'clean(base)', 'Narcissus', 'ICIT'))
    for arch, path in proxies.items():
        try:
            m = load_proxy(path, arch, 10, DEV)
        except Exception as e:
            print('%-10s (proxy missing: %s)' % (arch, e)); continue
        base = flip_rate(m, imgs, target, clean_apply) * 100
        nar = flip_rate(m, imgs, target, nar_apply) * 100
        icit = flip_rate(m, imgs, target, icit_apply) * 100
        print('%-10s %10.1f %12.1f %12.1f' % (arch, base, nar, icit))


if __name__ == '__main__':
    main()
