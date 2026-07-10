"""ICAF alpha-sweep + OPAL correctness smoke (no victim training)."""
import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.icaf_trigger import SmoothMask, isophote_field, icaf_apply, optimize_icaf_mask
from faat.opal_trigger import make_key, opal_project, opal_score

DEV = 'cuda'
torch.backends.cudnn.benchmark = False

ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
N = 3000
imgs = torch.stack([ds[i][0] for i in range(N)]).to(DEV)
labels = torch.tensor([ds[i][1] for i in range(N)], device=DEV)

proxy = ResNet18(num_classes=10).to(DEV)
ck = torch.load('resource/faat/proxy/resnet18_clean_cifar10.pth', map_location=DEV)
proxy.load_state_dict(ck['state_dict']); proxy.eval()
for p in proxy.parameters():
    p.requires_grad_(False)

def stealth(xa, xb):
    return (xa - xb).abs().max().item(), (xa - xb).abs().mean().item()

# ---------- ICAF alpha sweep ----------
print('=== ICAF alpha sweep (proxyASR vs stealth) ===')
for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
    info = optimize_icaf_mask(proxy, imgs, labels, target=0, device=DEV, alpha=a,
                              mask_res=8, steps=600, log_every=9999)
    trained = info['mask']()
    xp = icaf_apply(imgs[:1000], trained, a)
    linf, md = stealth(xp, imgs[:1000])
    print('  alpha=%4.1fpx  proxyASR=%.3f  L_inf=%.4f  mean|d|=%.5f'
          % (a, info['final_proxy_asr'], linf, md))

# ---------- OPAL correctness + stealth (consistent key) ----------
print('\n=== OPAL projection (consistent key: satisfaction + stealth) ===')
for levels in ['binary', 'cont']:
    for block in [4, 8]:
        nb = (32 // block) ** 2
        for margin in [4.0 / 255, 8.0 / 255, 16.0 / 255]:
            U, V, C, level = make_key(nb, M=128, seed=1, levels=levels)
            key = (U, V, C)
            z, _ = opal_project(imgs[:512], key, margin=margin, block=block)
            sat = opal_score(z, key, margin, block=block)
            linf = (z - imgs[:512]).abs().max().item()
            md = (z - imgs[:512]).abs().mean().item()
            print('  %s block=%d margin=%3d/255  satisfied=%.3f  L_inf=%.4f  mean|d|=%.5f'
                  % (levels, block, int(margin * 255), sat, linf, md))
print('OK')
