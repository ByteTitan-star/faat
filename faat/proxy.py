"""FAAT Stage B proxy model + target-class feature centre + per-sample summaries.

The proxy is a *clean* ResNet18 (trained on unpoisoned data). Its penultimate
features (512-d, via ResNet.extract_feature) define the alignment geometry:
    c_target = mean_{x in clean target class} f(x)
and L_align pulls poisoned x' toward c_target. Per plan 9.7.3 we train and save
the proxy here -- cal_metric.py is left untouched (it only writes .pkl).

select_poison_inds() mirrors train_backdoor.py exactly so that the artifacts
written by optimisation match the indices the host pipeline will inject.
"""
import os
import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from utils import get_stats, set_random_seed
from .strategy_net import compute_summary, N_BANDS


# --------------------------------------------------------------------------- #
# Poison-index selection -- identical to train_backdoor.py (do not diverge)
# --------------------------------------------------------------------------- #
def select_poison_inds(dataset, selection, output_dir, select_epoch, seed,
                       num_classes, y_target, res_sel, poison_rate, res_rate=1.0):
    total_poison = int(len(dataset) * poison_rate)
    if selection in ['loss', 'grad', 'forget', 'res', 'stealth']:
        stats_metric, stats_class, stats_inds = get_stats(
            selection, output_dir, select_epoch, seed, num_classes, y_target,
            res_sel, res_rate)
        metric_vals, metric_inds = [], []
        for i in range(len(dataset)):
            if stats_class[i] == y_target:
                metric_vals.append(stats_metric[i])
                metric_inds.append(stats_inds[i])
        if selection != 'stealth':
            largest = heapq.nlargest(total_poison, range(len(metric_vals)),
                                     metric_vals.__getitem__)
            return [metric_inds[i] for i in largest], total_poison
        return [], total_poison
    # random / poison
    shuffle = np.random.permutation(len(dataset))
    poison_inds, k = [], 0
    total_poison = len(dataset) * poison_rate
    for i in shuffle:
        if selection == 'poison':
            if dataset[i][1] != y_target and k < total_poison:
                poison_inds.append(int(i)); k += 1
        else:
            if dataset[i][1] == y_target and k < total_poison:
                poison_inds.append(int(i)); k += 1
    return poison_inds, total_poison


# --------------------------------------------------------------------------- #
# Proxy model
# --------------------------------------------------------------------------- #
def load_proxy(path, num_classes, device):
    net = ResNet18(num_classes=num_classes)
    if path and os.path.exists(path):
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        net.load_state_dict(sd)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train_clean_proxy(dataset='cifar10', num_classes=10, epochs=100,
                      lr=0.1, batch_size=128, device='cuda', save_path=None,
                      seed=1, data_dir='./data'):
    """Train a clean ResNet18 (no poison) and save its state_dict. Reuses the
    Table-1 optimiser recipe (SGD/nesterov/wd5e-4, milestones[60,90])."""
    set_random_seed(seed)
    # CIFAR10 returns PIL images -> augment in PIL space directly (no ToPILImage).
    tf = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    if dataset == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=True, transform=tf, download=True)
    elif dataset == 'cifar100':
        ds = datasets.CIFAR100(root='./data100', train=True, transform=tf, download=True)
    else:
        ds = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    net = ResNet18(num_classes=num_classes).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 90], gamma=0.1)
    crit = nn.CrossEntropyLoss().to(device)
    for ep in range(epochs):
        net.train()
        tot, cor = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            cor += out.argmax(1).eq(y).sum().item(); tot += loss.item() * y.size(0)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print('[proxy] ep %03d  loss %.3f  acc %.3f' % (ep, tot / len(ds), cor / len(ds)))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'state_dict': net.state_dict(),
                    'model_name': 'resnet18', 'num_classes': num_classes,
                    'clean': True, 'epochs': epochs}, save_path)
        print('[proxy] saved -> %s' % save_path)
    return net


# --------------------------------------------------------------------------- #
# Feature centre + per-sample summaries
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_center(proxy, dataset, target, device, max_n=4000, batch_size=256):
    """c_target = mean penultimate feature over clean target-class samples."""
    imgs = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target]
    if len(imgs) > max_n:
        imgs = imgs[:max_n]
    feats = []
    for s in range(0, len(imgs), batch_size):
        x = torch.stack(imgs[s:s + batch_size]).to(device)
        feats.append(proxy.extract_feature(x).cpu())
    return torch.cat(feats, dim=0).mean(dim=0)   # [D]


def stack_poison_images(dataset, poison_inds):
    return torch.stack([dataset[i][0] for i in poison_inds])   # [N,3,H,W]


@torch.no_grad()
def build_summaries(proxy, x_poison, c_target, device, size=32,
                    n_bands=N_BANDS, batch_size=128):
    """Cache s_i for every poison candidate. x_poison: [N,3,H,W] -> S [N,d_s]."""
    band_idx = _band_idx_local(size, n_bands)
    c = c_target.to(device)
    out = []
    for s in range(0, len(x_poison), batch_size):
        x = x_poison[s:s + batch_size].to(device)
        out.append(compute_summary(proxy, x, c, band_idx.to(device), n_bands).cpu())
    return torch.cat(out, dim=0)


def _band_idx_local(size, n_bands):
    # local import to avoid pulling image_stats private name at module top twice
    from .image_stats import _band_index_map
    return _band_index_map(size, n_bands)
