"""RKT eval: SSIM stealth + full-image NC anomaly + frequency high-freq ratio + AC/SS, on a
trained RKT victim. Reuses _nc_probe NC logic (deterministic) and metrics.detection for AC/SS.

Run: python -m faat._rkt_eval --rdir results/rkt_cifar10_s0.7_s1 \
       --trigger resource/faat/rkt/cifar10_s0.7_k5_s1 --target 0
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.rkt_trigger import load_rkt_trigger
from metrics.detection import extract_features, detection_metrics

# cuDNN conv non-determinism swings NC anomaly run-to-run (observed 4.94 vs 1.12 on the same
# model). Force determinism so the NC score is reproducible.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEV = 'cuda'


def ssim_torch(x, y, win=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    w = torch.ones(3, 1, win, win, device=x.device) / (win * win)

    def box(t):
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode='reflect'), w, groups=3)

    mx, my = box(x), box(y)
    vx = box(x * x) - mx * mx
    vy = box(y * y) - my * my
    cxy = box(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean().item()


def hf_ratio(x, cutoff=0.5):
    """Fraction of FFT magnitude energy in the high-freq band (radius > cutoff*max)."""
    X = torch.fft.fftshift(torch.fft.fft2(x))
    H = x.shape[-1]
    yy = torch.arange(H, device=x.device).view(-1, 1) - H // 2
    xx = torch.arange(H, device=x.device).view(1, -1) - H // 2
    r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    hf = (r > cutoff * H / 2).float()
    mag = X.abs()
    return (mag[..., hf > 0].sum() / (mag.sum() + 1e-8)).item()


def full_image_nc(model, imgs, target, nc, steps=800, M=2000, lr=0.02, lam=0.01):
    norms = []
    for c in range(nc):
        delta = torch.zeros(3, 32, 32, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        g = torch.Generator(device='cpu').manual_seed(c)
        idx_pool = torch.randperm(len(imgs), generator=g)[:M]
        tgt = torch.full((128,), c, dtype=torch.long, device=DEV)
        for step in range(steps):
            idx = idx_pool[torch.randint(0, M, (128,), generator=g)]
            x = imgs[idx]
            loss = F.cross_entropy(model(torch.clamp(x + delta, 0, 1)), tgt) + lam * delta.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nm = delta.flatten().norm().item()
        norms.append(nm)
    norms = np.array(norms)
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-8
    anomaly = (med - norms) / mad
    return norms, anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rdir', required=True)
    ap.add_argument('--trigger', required=True)
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--nc', type=int, default=10)
    args = ap.parse_args()

    ck = torch.load(args.rdir + '/model_last.pth', map_location=DEV)
    model = ResNet18(num_classes=args.nc).to(DEV)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    trig = load_rkt_trigger(args.trigger, DEV)

    ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV)

    xs = imgs[:1000]
    xsp = trig(xs)
    ssim = ssim_torch(xs, xsp)
    hf_clean = hf_ratio(xs)
    hf_trig = hf_ratio(xsp)
    l2 = (xsp - xs).flatten(1).norm(dim=1).mean().item()
    print('[rkt-eval] SSIM=%.3f  L2(mean)=%.4f  HF_clean=%.3f  HF_trig=%.3f' % (ssim, l2, hf_clean, hf_trig))

    norms, anomaly = full_image_nc(model, imgs, args.target, args.nc)
    flagged = [c for c in range(args.nc) if anomaly[c] > 2.0]
    print('[rkt-eval] NC norms: ' + ' '.join('%.2f' % n for n in norms))
    print('[rkt-eval] NC anomaly[target=%d]=%.2f  flagged=%s' % (args.target, anomaly[args.target], flagged))

    pj = json.load(open(args.rdir + '/poison_inds.json'))
    poison_inds = set(int(i) for i in pj['poison_inds'])
    train_ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    pois = [int(i) for i in pj['poison_inds']][:500]
    clean_pool = [i for i in range(len(train_ds)) if i not in poison_inds]
    cleans = np.random.RandomState(0).choice(clean_pool, 2000, replace=False).tolist()
    all_imgs, is_pois, labs = [], [], []
    for i in pois:
        all_imgs.append(trig(train_ds[i][0].unsqueeze(0).to(DEV))[0].cpu())
        is_pois.append(1); labs.append(args.target)
    for i in cleans:
        all_imgs.append(train_ds[int(i)][0]); is_pois.append(0); labs.append(train_ds[int(i)][1])
    feats = extract_features(model, torch.stack(all_imgs).to(DEV), device=DEV, layer='extract_feature')
    det = detection_metrics(feats, np.array(is_pois), np.array(labs))
    print('[rkt-eval] AC_AUC=%.3f  SS_AUC=%.3f' % (det['AC_AUC'], det['SS_AUC']))

    json.dump({'SSIM': ssim, 'L2': l2, 'HF_clean': hf_clean, 'HF_trig': hf_trig,
               'NC_anomaly_target': float(anomaly[args.target]),
               'NC_flagged': [int(c) for c in flagged],
               'AC_AUC': det['AC_AUC'], 'SS_AUC': det['SS_AUC']},
              open(args.rdir + '/rkt_eval.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
