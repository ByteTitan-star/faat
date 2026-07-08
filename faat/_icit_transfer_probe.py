"""Transferable-ICIT probe (path 3 validation): train the generator on an ENSEMBLE of
surrogate architectures (ResNet18+34+50) instead of ResNet18 only, so it finds a
cross-architecture trigger direction. Tests whether transfer to ResNet50 (29% single-arch)
can be pulled up -- if yes, 'transferable invisible trigger' is a real path-3 contribution.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from cifar_resnet import ResNet18, ResNet34, ResNet50
from faat.ic_trigger import ICGenerator
from faat.global_trigger import proxy_argmax

DEV = 'cuda'
ARCH = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50}


def load_proxy(path, arch):
    ck = torch.load(path, map_location=DEV)
    m = ARCH[arch](num_classes=10).to(DEV); m.load_state_dict(ck['state_dict']); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def flip_rate(model, imgs, target, apply):
    return model(apply(imgs)).argmax(1).eq(target).float().mean().item()


def main():
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(8000)]).to(DEV)
    target = 0
    proxies = {
        'resnet18': load_proxy('resource/faat/proxy/resnet18_clean_cifar10.pth', 'resnet18'),
        'resnet34': load_proxy('resource/faat/proxy/resnet34_clean_cifar10.pth', 'resnet34'),
        'resnet50': load_proxy('resource/faat/proxy/resnet50_clean_cifar10.pth', 'resnet50'),
    }
    # non-target pool (use R18 labels)
    lbl = proxy_argmax(proxies['resnet18'], imgs, DEV)
    pool = torch.where(lbl != target)[0].to(DEV)
    imgs = imgs[pool]; M = len(imgs)
    tgt = torch.full((128,), target, dtype=torch.long, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(0)

    for mode, members in [('single-R18', ['resnet18']), ('ensemble-R18+34+50', ['resnet18', 'resnet34', 'resnet50'])]:
        gen = ICGenerator().to(DEV)
        opt = torch.optim.Adam(gen.parameters(), lr=5e-4)
        budget = 2.0
        for step in range(3000):
            idx = torch.randint(0, M, (128,), generator=g); x = imgs[idx]
            xp = torch.clamp(x + gen(x, budget), 0, 1)
            loss = sum(F.cross_entropy(proxies[a](xp), tgt) for a in members) / len(members)
            opt.zero_grad(); loss.backward(); opt.step()
        apply = lambda xx, _g=gen, _b=budget: torch.clamp(xx + _g(xx, _b), 0, 1)
        rates = {a: flip_rate(proxies[a], imgs[:1500], target, apply) * 100 for a in proxies}
        print('%-22s ' % mode + '  '.join('%s=%.0f%%' % (a, rates[a]) for a in rates))


if __name__ == '__main__':
    main()
