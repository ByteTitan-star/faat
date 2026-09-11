"""Black-box transfer eval (Phase 1 core): apply ensemble-ICIT trigger vs single-R18 ICIT
trigger to HELD-OUT architectures (ResNet101/152 -- NOT in the R18+34+50 ensemble) and
measure transfer ASR (flip rate to target). Shows ensemble training gives true black-box
transfer to unseen deeper architectures.
"""
import torch
from torchvision import datasets, transforms
from cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from faat.ic_trigger import load_ic_generator

DEV = 'cuda'
ARCH = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50,
        'resnet101': ResNet101, 'resnet152': ResNet152}


def load_clean(path, arch, nc=10):
    ck = torch.load(path, map_location=DEV)
    m = ARCH[arch](num_classes=nc).to(DEV); m.load_state_dict(ck['state_dict']); m.eval()
    return m


@torch.no_grad()
def flip_rate(model, imgs, target, apply):
    return model(apply(imgs)).argmax(1).eq(target).float().mean().item() * 100


def main():
    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(3000)]).to(DEV)
    target = 0

    ens_gen, ens_b = load_ic_generator('resource/faat/icit_ens/cifar10_b2.0_s1', DEV)
    sgl_gen, sgl_b = load_ic_generator('resource/faat/icit/cifar10_l2_2.0_seed1', DEV)
    ens_apply = lambda x: torch.clamp(x + ens_gen(x, ens_b), 0, 1)
    sgl_apply = lambda x: torch.clamp(x + sgl_gen(x, sgl_b), 0, 1)
    clean_apply = lambda x: x

    targets = [('resnet18', 'resource/faat/proxy/resnet18_clean_cifar10.pth'),
               ('resnet101', 'resource/faat/proxy/resnet101_clean_cifar10.pth'),
               ('resnet152', 'resource/faat/proxy/resnet152_clean_cifar10.pth')]
    print('%-12s %10s %14s %14s' % ('target arch', 'clean(base)', 'single-R18 ICIT', 'ensemble-ICIT'))
    for arch, path in targets:
        try:
            m = load_clean(path, arch)
        except Exception as e:
            print('%-12s (clean model missing: %s)' % (arch, e)); continue
        base = flip_rate(m, imgs, target, clean_apply)
        sgl = flip_rate(m, imgs, target, sgl_apply)
        ens = flip_rate(m, imgs, target, ens_apply)
        star = '  <- held-out' if arch in ('resnet101', 'resnet152') else ''
        print('%-12s %10.1f %14.1f %14.1f%s' % (arch, base, sgl, ens, star))


if __name__ == '__main__':
    main()
