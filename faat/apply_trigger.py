"""Stage A FAAT eager injection.

Mirrors Add_Clean_Label_Train_Trigger_NAR (_siba) / Add_Test_Trigger_NAR in utils.py,
returning the same tuple shapes the host pipeline (MyDataset / DataLoaders) expects.

  train: x'_i = clamp(x_i + global_scale*delta_global + delta_adaptive_i, 0, 1),
         poisoned samples relabeled to `target` (host-repo "clean-label" convention:
         relabel to target; see docs/plans.md §0). delta_adaptive_i is rule-based,
         train-only, and depends on the sample's image statistics.
  test : x'_test = clamp(x_test + global_scale*delta_global, 0, 1)   (C2: index-independent)

delta_global is the Narcissus noise (a validated clean-label direction); it is the
workhorse that carries the attack and the ONLY thing applied at test time. delta_adaptive
is the FAAT-specific sample-adaptive residual being validated.
"""
import os

import numpy as np
import torch

from .rules import rule_params, generate_adaptive_delta

NAR_NOISE = './resource/narcissus/noise_01000.pth'


def _load_global_delta(save_trigger, global_scale):
    """Load (or cache) delta_global = Narcissus noise, scaled by global_scale -> [3,H,W]."""
    os.makedirs(save_trigger, exist_ok=True)
    gpath = os.path.join(save_trigger, 'global_delta.npy')
    if os.path.exists(gpath):
        delta = torch.from_numpy(np.load(gpath))
    elif os.path.exists(NAR_NOISE):
        t = torch.load(NAR_NOISE, map_location='cpu')
        delta = t.squeeze(0).float()
        np.save(gpath, delta.numpy())
    else:
        # fallback: deterministic low-freq-ish noise if Narcissus artifact is absent
        g = torch.Generator().manual_seed(12345)
        delta = 0.05 * torch.randn((3, 32, 32), generator=g)
        np.save(gpath, delta.numpy())
    return (delta * float(global_scale)).float()


def Add_Clean_Label_Train_Trigger_faat(dataset, target, class_order, save_trigger,
                                       global_scale=1.0, eps=None):
    delta_global = _load_global_delta(save_trigger, global_scale)
    poison_set = set(int(i) for i in class_order)
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        if i in poison_set:
            band_weights, eps_i = rule_params(img)
            if eps is not None:
                eps_i = float(eps)
            delta_adaptive = generate_adaptive_delta(img.float(), band_weights, eps_i, seed=i)
            temp = torch.clamp(img.float() + delta_global + delta_adaptive, 0.0, 1.0)
            dataset_.append((temp, target, 1))
        else:
            dataset_.append((img, data[1], 0))
    return dataset_


def Add_Test_Trigger_faat(dataset, target, save_trigger, global_scale=1.0):
    delta_global = _load_global_delta(save_trigger, global_scale)
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:        # match host convention: skip target-class test images
            continue
        temp = torch.clamp(img.float() + delta_global, 0.0, 1.0)
        dataset_.append((temp, target))
    return dataset_


# --------------------------------------------------------------------------- #
# Stage B (learned differentiable policy + feature alignment) injection.
# Artifacts come from faat/optimize.py:
#   global_delta.npy [3,H,W], adaptive_delta.npy [N,3,H,W], poison_keys.npy [N].
# Train: x'_i = clamp(x_i + gs*delta_global + delta_adaptive_i, 0,1)  (poisoned only)
# Test : x'_t = clamp(x_t   + gs*delta_global,            0,1)         (C2: global only)
# --------------------------------------------------------------------------- #
def _load_faatb_artifacts(save_trigger, global_scale=1.0):
    gpath = os.path.join(save_trigger, 'global_delta.npy')
    apath = os.path.join(save_trigger, 'adaptive_delta.npy')
    kpath = os.path.join(save_trigger, 'poison_keys.npy')
    delta_global = torch.from_numpy(np.load(gpath)).float() * float(global_scale)
    adaptive_map = {}
    if os.path.exists(apath) and os.path.exists(kpath):
        adaptive = np.load(apath)                 # [N,3,H,W]
        keys = np.load(kpath)                     # [N]
        for row, k in enumerate(keys):
            adaptive_map[int(k)] = torch.from_numpy(adaptive[row]).float()
    return delta_global, adaptive_map


def Add_Clean_Label_Train_Trigger_faatb(dataset, target, poison_inds, save_trigger,
                                        global_scale=1.0):
    delta_global, adaptive_map = _load_faatb_artifacts(save_trigger, global_scale)
    poison_set = set(int(i) for i in poison_inds)
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        if i in poison_set:
            da = adaptive_map.get(int(i))         # None if a key is missing -> global only
            temp = img.float() + delta_global
            if da is not None:
                temp = temp + da
            temp = torch.clamp(temp, 0.0, 1.0)
            dataset_.append((temp, target, 1))
        else:
            dataset_.append((img, data[1], 0))
    return dataset_


def Add_Test_Trigger_faatb(dataset, target, save_trigger, global_scale=1.0):
    delta_global, _ = _load_faatb_artifacts(save_trigger, global_scale)
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        temp = torch.clamp(img.float() + delta_global, 0.0, 1.0)
        dataset_.append((temp, target))
    return dataset_
