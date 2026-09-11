"""Input-Conditioned imperceptible trigger probe (NC-evasion hypothesis).

A small generator g(x) produces a per-image BOUNDED perturbation. Trained (frozen proxy)
so proxy(clamp(x+g(x),0,1)) -> target. g is SHARED (index-independent -> satisfies C2),
but the effective trigger g(x) VARIES per image -> NC's universal-delta reverse-engineering
cannot capture it with one small delta per class -> evades NC (constructively).

Probe: can g(x) achieve high proxy-ASR while imperceptible (high SSIM, low L-inf)?
If yes, the NC-evasion attack path is viable (then confirm with victim training + NC).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from faat.proxy import load_proxy
from faat.global_trigger import proxy_argmax

DEV = 'cuda'


def ssim_torch(x, y, win=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    w = torch.ones(3, 1, win, win, device=x.device) / (win * win)

    def box(t):
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode='reflect'), w, groups=3)

    mu_x, mu_y = box(x), box(y)
    vx = box(x * x) - mu_x * mu_x
    vy = box(y * y) - mu_y * mu_y
    cxy = box(x * y) - mu_x * mu_y
    return (((2 * mu_x * mu_y + C1) * (2 * cxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (vx + vy + C2))).mean().item()


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x, budget):
        d = self.net(x)
        # per-image L2 projection to <= budget (like Narcissus, but per-image)
        flat = d.flatten(1)
        n = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(budget / n, max=1.0)
        return d * scale.view(-1, 1, 1, 1)


def main():
    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(8000)]).to(DEV)
    proxy = load_proxy('resource/faat/proxy/resnet18_clean_cifar10.pth', 10, DEV)
    for p in proxy.parameters():
        p.requires_grad_(False)
    lbl = proxy_argmax(proxy, imgs, DEV)
    pool = torch.where(lbl != 0)[0].to(DEV)
    imgs = imgs[pool]
    M = len(imgs)
    tgt = torch.full((128,), 0, dtype=torch.long, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(0)

    print('%-8s %8s %8s %8s %8s' % ('budget', 'proxyASR', 'SSIM', 'Linf/255', 'meanD/255'))
    for budget in [1.0, 1.5, 2.0, 3.0]:
        gen = Gen().to(DEV)
        opt = torch.optim.Adam(gen.parameters(), lr=5e-4)
        for step in range(3000):
            idx = torch.randint(0, M, (128,), generator=g)
            x = imgs[idx]
            d = gen(x, budget)
            xp = torch.clamp(x + d, 0, 1)
            loss = F.cross_entropy(proxy(xp), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x = imgs[:1000]
            d = gen(x, budget); xp = torch.clamp(x + d, 0, 1)
            asr = proxy(xp).argmax(1).eq(0).float().mean().item()
            print('%-8s %8.3f %8.3f %8.1f %8.2f' % (
                budget, asr, ssim_torch(x, xp),
                d.abs().max().item() * 255, d.abs().mean().item() * 255))


if __name__ == '__main__':
    main()
