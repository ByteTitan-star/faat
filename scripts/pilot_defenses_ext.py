"""Defense-layer extension for representative OOD-ABC checkpoints (RQ4).

Completes the RQ4 evidence chain beyond AC/SS (already computed) with:
  STRIP  — test-input detection  : AUC + TPR@FPR=5% (lower entropy = backdoored)
  NC     — model/class anomaly   : anomaly index = L1(mask_target) / median L1(mask_other)
                                   (<2 conventionally reads as evaded; Wang et al. 2019)
  FP     — mitigation            : ASR / BA after pruning 90% of least-active
                                   channels in layer4.1.conv2 (fine-pruning)

Functions mirror the frozen repo's faat/defenses.py (read-only reference);
self-contained here to avoid cross-repo imports.

Representative subset = main-grid runs (no _ep150/_pr/_w variants):
GTSRB a/b/c/cur x 2 seeds (c = non-acquired), CIFAR-10 + CIFAR-100 main arms.

Output: docs/pilot_defenses_ext.csv (joins 1:1 with docs/pilot_metrics.csv by tag).

Usage:  CUDA_VISIBLE_DEVICES=<gpu> python scripts/pilot_defenses_ext.py --device cuda [--tags t1 t2 ...]
"""
import os
import sys
import glob
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_resnet import ResNet18
from scripts.pilot_metrics import load_test


# --------------------------------------------------------------------------- #
# STRIP (faat/defenses.py:strip_scores + detection, self-contained)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def strip_scores(model, x, device, n_perturb=50, noise_std=0.03):
    B, C, H, W = x.shape
    x = x.to(device)
    ent = torch.zeros(B, device=device)
    for _ in range(n_perturb):
        x_p = torch.clamp(x + torch.randn(B, C, H, W, device=device) * noise_std, 0.0, 1.0)
        probs = F.softmax(model(x_p), dim=1)
        ent += -(probs * (probs + 1e-9).log()).sum(dim=1)
    return (ent / n_perturb).cpu().numpy()


def _auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]; P = labels.sum(); N = len(labels) - P
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tp / P]); fpr = np.concatenate([[0.0], fp / N])
    return float(np.trapz(tpr, fpr))


def _tpr_at_fpr(scores, labels, fpr_target=0.05):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return float('nan')
    order = np.argsort(-scores, kind='stable'); labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / P; fpr = fp / N
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0


def strip_eval(model, x_clean, x_trig, device, n=500, n_perturb=50):
    rng = np.random.RandomState(0)
    ic = rng.choice(len(x_clean), min(n, len(x_clean)), replace=False)
    ip = rng.choice(len(x_trig), min(n, len(x_trig)), replace=False)
    s_p = strip_scores(model, x_trig[ip], device, n_perturb)
    s_c = strip_scores(model, x_clean[ic], device, n_perturb)
    all_s = -np.concatenate([s_p, s_c])          # anomaly = -entropy
    labels = np.array([1] * len(s_p) + [0] * len(s_c))
    return _auc(all_s, labels), _tpr_at_fpr(all_s, labels, 0.05)


# --------------------------------------------------------------------------- #
# Neural Cleanse (Wang et al. 2019) — mask+pattern inversion per class
# --------------------------------------------------------------------------- #
def nc_anomaly_index(model, x_pool, target, device, n_classes,
                     n_steps=80, batch=200, lam=0.01, n_input=1000, seed=0):
    """Reverse-engineer a minimal trigger per class; anomaly = target L1 vs median others."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x_pool), min(n_input, len(x_pool)), replace=False)
    xs = x_pool[idx]                                  # stays on CPU; batched to device
    g = torch.Generator().manual_seed(seed)

    def invert(cls):
        mask = torch.zeros(1, 1, 32, 32, device=device, requires_grad=True)
        patt = torch.zeros(3, 32, 32, device=device, requires_grad=True)
        opt = torch.optim.Adam([mask, patt], lr=0.1)
        for _ in range(n_steps):
            bidx = torch.randint(0, len(xs), (min(batch, len(xs)),), generator=g)
            xb = xs[bidx].to(device)
            m = torch.clamp(torch.sigmoid(mask), 0, 1)
            p = torch.clamp(patt, 0, 1)
            xt = (1 - m) * xb + m * p
            loss = F.cross_entropy(model(xt), torch.full((len(xb),), cls, device=device,
                                                         dtype=torch.long)) + lam * m.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.clamp(torch.sigmoid(mask), 0, 1).abs().sum().item()

    l1s = {}
    for c in range(n_classes):
        l1s[c] = invert(c)
    others = np.median([v for c, v in l1s.items() if c != target])
    return l1s[target] / max(others, 1e-6), l1s[target], others


# --------------------------------------------------------------------------- #
# Fine-Pruning (faat/defenses.py:fine_pruning_defense, 90% point)
# --------------------------------------------------------------------------- #
def fp_eval(model, x_clean, y_clean, x_trig, target, device,
            layer_name='layer4.1.conv2', prune_ratio=0.9, n_prof=2000):
    import copy
    model = copy.deepcopy(model)
    # profile activations on clean images (hook MUST return None — a non-None
    # return would replace the module output with the [C] activation vector)
    store = {}
    def _hook(m, i, o):
        store['a'] = o.mean(dim=(0, 2, 3)).detach()
    h = dict(model.named_modules())[layer_name].register_forward_hook(_hook)
    with torch.no_grad():
        for s in range(0, min(n_prof, len(x_clean)), 256):
            model(x_clean[s:s + 256].to(device))
    h.remove()
    acts = store['a'].cpu().numpy()
    order = np.argsort(acts)                       # least active first
    conv = dict(model.named_modules())[layer_name]
    n_prune = int(len(order) * prune_ratio)
    with torch.no_grad():
        idx = torch.tensor(order[:n_prune], device=device)
        conv.weight[idx] = 0
        if conv.bias is not None:
            conv.bias[idx] = 0
    # eval (batched — full-set single forward OOMs on layer1 activations)
    model.eval()
    correct_p = total = correct_c = 0
    with torch.no_grad():
        for s in range(0, len(x_trig), 256):
            out = model(x_trig[s:s + 256].to(device))
            correct_p += (out.argmax(1).cpu() == target).sum().item()
        total = len(x_trig)
        for s in range(0, len(x_clean), 256):
            out = model(x_clean[s:s + 256].to(device))
            correct_c += (out.argmax(1).cpu() == y_clean[s:s + 256]).sum().item()
    return 100 * correct_p / max(total, 1), 100 * correct_c / len(x_clean)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def asr_of(model, x_trig, target, device, bs=512):
    correct = 0
    for s in range(0, len(x_trig), bs):
        out = model(x_trig[s:s + bs].to(device))
        correct += (out.argmax(1).cpu() == target).sum().item()
    return 100 * correct / len(x_trig)


@torch.no_grad()
def ba_of(model, x_clean, y_clean, device, bs=512):
    correct = 0
    for s in range(0, len(x_clean), bs):
        out = model(x_clean[s:s + bs].to(device))
        correct += (out.argmax(1).cpu() == y_clean[s:s + bs]).sum().item()
    return 100 * correct / len(x_clean)


def load_test_with_labels(dataset, data_dir):
    """Full test set with labels (BA reference). Mirrors pilot_metrics.load_test transforms."""
    from torchvision import datasets, transforms
    if dataset == 'cifar10':
        ds = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
    elif dataset == 'cifar100':
        ds = datasets.CIFAR100('./data100', train=False, transform=transforms.ToTensor())
    else:
        ds = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                  transform=transforms.Compose([transforms.Resize(32),
                                                                transforms.ToTensor()]))
    xs, ys = [], []
    for i in range(len(ds)):
        xs.append(ds[i][0]); ys.append(int(ds[i][1]))
    return torch.stack(xs), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--nc_steps', type=int, default=80)
    ap.add_argument('--tags', nargs='*', default=None,
                    help='explicit run tags (default: representative main-grid subset)')
    args = ap.parse_args()
    dev = args.device

    # representative main-grid tags
    tags = []
    for ds, seeds in [('gtsrb', [1, 2]), ('cifar10', [1, 2]), ('cifar100', [1])]:
        for arm in ['a', 'b', 'c', 'cur']:
            for s in seeds:
                t = f'oodabc_{ds}_{arm}_seed{s}'
                if os.path.exists(os.path.join('results_ood', t, 'model_last.pth')):
                    tags.append(t)
    if args.tags:
        tags = args.tags

    out_rows = []
    for t in tags:
        rdir = os.path.join('results_ood', t)
        tdir = os.path.join('resource_ood', 'triggers', t)
        dg = torch.from_numpy(np.load(os.path.join(tdir, 'global_delta.npy'))).float().to(dev)
        ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
        ds = 'cifar100' if 'cifar100' in t else ('cifar10' if 'cifar10' in t else 'gtsrb')
        n_cls = 100 if ds == 'cifar100' else (10 if ds == 'cifar10' else 43)
        model = ResNet18(num_classes=n_cls)
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()

        xt, xnt, nc, yt = load_test(ds, 'data/GTSRB32' if ds == 'gtsrb' else './data', dev)
        x_all, y_all = load_test_with_labels(ds, 'data/GTSRB32' if ds == 'gtsrb' else './data')
        target = 0
        x_trig = torch.clamp(xnt + dg.unsqueeze(0), 0.0, 1.0)

        asr0 = asr_of(model, x_trig, target, dev)
        ba0 = ba_of(model, x_all, y_all, dev)

        # STRIP
        strip_auc, strip_tpr = strip_eval(model, xt, x_trig, dev)
        # NC (all classes inversion)
        nc_idx, l1_t, l1_med = nc_anomaly_index(model, xnt, target, dev, n_cls,
                                                n_steps=args.nc_steps)
        # FP (90% prune; destructive -> run last, on a deepcopy inside fp_eval)
        fp_asr, fp_ba = fp_eval(model, x_all, y_all, x_trig, target, dev)

        row = dict(tag=t, dataset=ds, arm=t.split('_')[2], seed=t.split('_')[3],
                   asr=round(asr0, 2), ba=round(ba0, 2),
                   strip_auc=round(strip_auc, 4), strip_tpr5=round(strip_tpr, 4),
                   nc_anom_idx=round(nc_idx, 3), nc_l1_target=round(l1_t, 2),
                   nc_l1_median=round(l1_med, 2),
                   fp90_asr=round(fp_asr, 2), fp90_ba=round(fp_ba, 2))
        out_rows.append(row)
        print('[def] %-30s ASR=%5.1f | STRIP auc=%.3f tpr5=%.3f | NC idx=%.2f | FP90 asr=%5.1f ba=%5.1f'
              % (t, asr0, strip_auc, strip_tpr, nc_idx, fp_asr, fp_ba), flush=True)
        del model
        torch.cuda.empty_cache()

    out_csv = 'docs/pilot_defenses_ext.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print('wrote %s (%d rows)' % (out_csv, len(out_rows)))


if __name__ == '__main__':
    main()
