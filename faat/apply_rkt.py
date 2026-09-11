"""RKT eager injection: apply the (frozen, shared) resampling trigger to poison train
samples and to all non-target test images.

  train: x'_i = RKT(x_i)  for poison samples (relabeled to target); clean samples untouched.
         (host "clean-label" convention: relabel poison to target.)
  test : x'_test = RKT(x_test)  (C2 OK: (s,K) is shared & index-independent -- the trigger
         is a fixed global resampling, unlike ICIT's per-image generator.)
"""
import torch

from .rkt_trigger import load_rkt_trigger


def Add_Clean_Label_Train_Trigger_rkt(dataset, target, poison_inds, save_trigger, device='cuda'):
    trig = load_rkt_trigger(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            if i in poison_set:
                img_p = trig(img.unsqueeze(0).to(device))[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((img, data[1], 0))
    return out


def Add_Test_Trigger_rkt(dataset, target, save_trigger, device='cuda'):
    trig = load_rkt_trigger(save_trigger, device)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            label = data[1]
            if label == target:        # match host convention: skip target-class test images
                continue
            img_p = trig(img.unsqueeze(0).to(device))[0].cpu()
            out.append((img_p, target))
    return out
