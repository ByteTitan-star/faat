"""In-repo lightweight defenses (STRIP + Fine-Pruning), complementing the
already-implemented AC + SS in metrics/detection.py. Each defense is a function
that takes a trained victim model and returns detection or mitigation metrics.

  * STRIP : perturbation-consistency detection. Under random input perturbations,
           a backdoored model's prediction for triggered images stays abnormally
           stable (low entropy). Reports TPR@FPR=5% on distinguishing poisoned
           test images from clean.
  * Fine-Pruning : prunes the least-active neurons (on clean data) and measures
           the ASR/BA trade-off as a function of prune rate. A successful defense
           causes ASR to fall before BA does.

Run: python -m faat.defenses --model results/faatb_v3_1_scale_0.2/model_last.pth \
     --poison_inds results/faatb_v3_1_scale_0.2/poison_inds.json \
     --delta_global resource/faat/v3_1/scale_0.2/global_delta.npy --device cuda
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cifar_resnet import ResNet18


# --------------------------------------------------------------------------- #
# STRIP — perturbation-consistency detection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def strip_scores(model, x, device, n_perturb=50, noise_std=0.03):
    """STRIP anomaly score (lower = more suspicious = backdoored) for each image.
    x: [N,3,H,W] batch of images (triggered). Returns [N] scores."""
    B, C, H, W = x.shape
    x = x.to(device)
    entropy_sum = torch.zeros(B, device=device)
    for _ in range(n_perturb):
        noise = torch.randn(B, C, H, W, device=device) * noise_std
        x_p = torch.clamp(x + noise, 0.0, 1.0)
        probs = F.softmax(model(x_p), dim=1)
        entropy_sum += -(probs * (probs + 1e-9).log()).sum(dim=1)
    return (entropy_sum / n_perturb).cpu().numpy()


def strip_detection(model, test_dataset, target, device, n_clean=500, n_perturb=50):
    """STRIP detection: TPR@FPR=5% separating poison from clean test images."""
    # poison images (non-target, +trigger applied)
    dg = torch.from_numpy(np.load(model._delta_global_path)).float() \
        if hasattr(model, '_delta_global_path') else torch.zeros(3, 32, 32)
    dg = dg.to(device)
    non_t = [i for i in range(len(test_dataset)) if test_dataset[i][1] != target]
    n_pois = min(n_clean, len(non_t))
    idx_p = np.random.RandomState(0).choice(non_t, n_pois, replace=False)
    x_p = torch.clamp(torch.stack([test_dataset[int(i)][0] for i in idx_p]).to(device) + dg, 0, 1)
    # clean images
    idx_c = np.random.RandomState(0).choice(len(test_dataset), n_clean, replace=False)
    x_c = torch.stack([test_dataset[int(i)][0] for i in idx_c]).to(device)
    scores_p = strip_scores(model, x_p, device, n_perturb)
    scores_c = strip_scores(model, x_c, device, n_perturb)
    # Lower STRIP entropy → trigger → backdoor. Anomaly score = -entropy (higher = backdoor)
    all_scores = -np.concatenate([scores_p, scores_c])
    labels = np.array([1] * len(scores_p) + [0] * len(scores_c))
    return _auc(all_scores, labels), _tpr_at_fpr(all_scores, labels, 0.05)


# --------------------------------------------------------------------------- #
# Fine-Pruning — neuron pruning defense
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _record_activations(model, loader, device, layer_name='layer4.1.conv2'):
    """Record mean activation of each output channel in the target conv layer
    using a forward hook (standard FP approach)."""
    activations = []
    def hook(m, inp, outp):
        activations.append(outp.detach().mean(dim=(2, 3)).mean(dim=0).cpu())
    handle = None
    for name, m in model.named_modules():
        if name == layer_name:
            handle = m.register_forward_hook(hook)
            break
    if handle is None:
        raise ValueError('layer %s not found' % layer_name)
    model.eval()
    for x, _ in loader:
        model(x.to(device))
    handle.remove()
    return torch.stack(activations).mean(dim=0)   # [channels]


@torch.no_grad()
def _prune_channels(model, layer_name, prune_mask):
    """Zero out output channels in the target conv layer."""
    for name, m in model.named_modules():
        if name == layer_name:
            m.weight[prune_mask] = 0.0
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias[prune_mask] = 0.0
    return model


def _eval_asr_ba(model, clean_loader, poison_loader, device):
    model.eval()
    clean_correct, clean_total = 0, 0
    poison_correct, poison_total = 0, 0
    for x, y in clean_loader:
        out = model(x.to(device))
        clean_correct += (out.argmax(1).cpu() == y).sum().item()
        clean_total += y.size(0)
    for x, y in poison_loader:
        out = model(x.to(device))
        poison_correct += (out.argmax(1).cpu() == y).sum().item()
        poison_total += y.size(0)
    return 100 * poison_correct / max(poison_total, 1), 100 * clean_correct / max(clean_total, 1)


def fine_pruning_defense(model, clean_train, poison_test, target, device,
                         prune_ratios=None, layer_name='layer4.1.conv2'):
    """Prune least-active channels in `layer_name` and record ASR/BA curve."""
    if prune_ratios is None:
        prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    # small clean loader for activation profiling
    n_prof = min(2000, len(clean_train))
    sampler = np.random.RandomState(0).choice(len(clean_train), n_prof, replace=False)
    prof_loader = DataLoader(Subset(clean_train, sampler), batch_size=128, shuffle=False)
    acts = _record_activations(model, prof_loader, device, layer_name)
    order = acts.argsort()  # least active first
    # poison loader for ASR eval
    pois_ds = [(torch.zeros(3, 32, 32), 0)]  # placeholder; will build below
    # We'll construct the poison loader from test data + trigger
    dg = torch.from_numpy(np.load(getattr(model, '_delta_global_path', 'nonexist'))).float() \
        if hasattr(model, '_delta_global_path') and os.path.exists(getattr(model, '_delta_global_path', '')) \
        else torch.zeros(3, 32, 32)
    # poison test = non-target images + trigger, relabeled target
    non_t = [i for i in range(len(poison_test)) if poison_test[i][1] != target]
    pois = torch.stack([torch.clamp(poison_test[int(i)][0] + dg, 0, 1) for i in non_t[:1000]])
    p_labels = torch.full((len(pois),), target, dtype=torch.long)
    poison_loader = DataLoader(list(zip(pois, p_labels)), batch_size=128, shuffle=False)
    clean_loader = DataLoader(poison_test, batch_size=128, shuffle=False)
    curve = []
    for r in prune_ratios:
        n_prune = int(len(order) * r)
        if n_prune > 0:
            prune_mask = order[:n_prune]
            _prune_channels(model, layer_name, prune_mask)
        asr, ba = _eval_asr_ba(model, clean_loader, poison_loader, device)
        curve.append({'prune_ratio': float(r), 'ASR': round(asr, 2), 'BA': round(ba, 2)})
    return curve


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _auc(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]; P = labels.sum(); N = len(labels) - P
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0.0], tp / P]); fpr = np.concatenate([[0.0], fp / N])
    return float(np.trapz(tpr, fpr))


def _tpr_at_fpr(scores, labels, fpr_target=0.05):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0: return float('nan')
    order = np.argsort(-scores, kind='stable'); labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / P; fpr = fp / N
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser('FAAT in-repo defenses (STRIP + FP)')
    ap.add_argument('--model_dir', default='results/faatb_v3_1_scale_0.2')
    ap.add_argument('--delta_global', default='resource/faat/v3_1/scale_0.2/global_delta.npy')
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    ds = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=False)
    test_ds = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
    pj = json.load(open(os.path.join(args.model_dir, 'poison_inds.json')))
    ck = torch.load(os.path.join(args.model_dir, 'model_last.pth'), map_location='cpu')
    model = ResNet18(num_classes=int(ck.get('num_classes', 10)))
    model.load_state_dict(ck['state_dict'])
    model._delta_global_path = args.delta_global
    model = model.to(dev).eval()

    print('=== STRIP ===')
    strip_auc, strip_tpr5 = strip_detection(model, test_ds, args.y_target, dev)
    print('STRIP AUC=%.3f TPR@5%%=%.3f (低=攻击规避STRIP)' % (strip_auc, strip_tpr5))

    print('=== Fine-Pruning ===')
    curve = fine_pruning_defense(model, ds, test_ds, args.y_target, dev)
    for pt in curve:
        print('  prune=%.1f ASR=%.1f BA=%.1f' % (pt['prune_ratio'], pt['ASR'], pt['BA']))

    # write JSON
    out = {'STRIP': {'AUC': round(strip_auc, 3), 'TPR@5FPR': round(strip_tpr5, 3)},
           'FinePruning': curve}
    with open(os.path.join(args.model_dir, 'defenses.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote %s/defenses.json' % args.model_dir)


if __name__ == '__main__':
    main()
