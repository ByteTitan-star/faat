"""Neural Cleanse probe: does NC detect a Narcissus-style invisible backdoor?

For each class c, reverse-engineer the smallest all-pixel L2 perturbation delta_c that
flips clean images -> c under the backdoored model. The backdoor TARGET class needs a
markedly smaller delta_c (the backdoor does the flipping for free). NC flags a class whose
trigger norm is anomalously small (anomaly index > 2).

This is the all-pixel L2 variant of NC (appropriate for invisible additive triggers;
classic NC uses a patch mask + TV + L1, which assumes patch triggers). If this detects
the target class, then Neural-Cleanse-class recovery defenses CATCH Narcissus -> there is
attack headroom (design a trigger that evades NC). If not, the space is saturated.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from cifar_resnet import ResNet18

DEV = 'cuda'
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else 'results/faatb_v4_cifar10_l2_1.5_seed1'
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def main():
    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)  # test set clean
    ck = torch.load(MODEL_DIR + '/model_last.pth', map_location=DEV)
    nc = int(ck.get('num_classes', 10))
    model = ResNet18(num_classes=nc).to(DEV)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    print('[nc] model %s, target=%d, %d clean test imgs' % (MODEL_DIR, TARGET, len(imgs)))

    norms = []
    for c in range(nc):
        delta = torch.zeros(3, 32, 32, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.02)
        lam = 0.01  # L2 penalty weight (NC keeps trigger small)
        g = torch.Generator(device='cpu').manual_seed(c)
        M = 2000
        idx_pool = torch.randperm(len(imgs), generator=g)[:M]
        tgt = torch.full((128,), c, dtype=torch.long, device=DEV)
        for step in range(800):
            idx = idx_pool[torch.randint(0, M, (128,), generator=g)]
            x = imgs[idx]
            loss = F.cross_entropy(model(torch.clamp(x + delta, 0, 1)), tgt) + lam * delta.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            # flip rate to c at final delta
            fr = 0; n = 0
            for s in range(0, M, 512):
                out = model(torch.clamp(imgs[idx_pool[s:s + 512]] + delta, 0, 1))
                fr += out.argmax(1).eq(c).sum().item(); n += out.size(0)
            fr /= n
            nm = delta.flatten().norm().item()
        norms.append(nm)
        flag = '  <== TARGET' if c == TARGET else ''
        print('[nc] class %2d: ||delta||=%.3f  flipRate=%.3f%s' % (c, nm, fr, flag))
    norms = np.array(norms)
    # NC anomaly index: class with anomalously SMALL norm (using median/MAD)
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-8
    anomaly = (med - norms) / mad  # small norm -> high positive anomaly
    print('\n[nc] median ||delta||=%.3f  MAD=%.3f' % (med, mad))
    print('[nc] anomaly index per class (small-norm -> high):')
    for c in range(nc):
        print('       class %2d: %.2f%s' % (c, anomaly[c], '  <== TARGET' if c == TARGET else ''))
    flagged = [c for c in range(nc) if anomaly[c] > 2.0]
    hit = TARGET in flagged
    print('\n[nc] NC flags classes with anomaly>2: %s' % (flagged if flagged else 'NONE'))
    print('[nc] *** NC %s the Narcissus backdoor (target %d flagged=%s) ***'
          % ('CATCHES' if hit else 'MISSES', TARGET, hit))


if __name__ == '__main__':
    main()
