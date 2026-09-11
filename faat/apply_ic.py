"""ICIT eager injection: apply the (frozen, shared) input-conditioned generator g(x) to
poison train samples and to all non-target test images.

  train: x'_i = clamp(x_i + g(x_i, budget), 0, 1)  for poison samples (relabeled to target);
         clean samples untouched. (host "clean-label" convention: relabel poison to target.)
  test : x'_test = clamp(x_test + g(x_test, budget), 0, 1)  (C2 OK: g is shared; only the
         per-image application is input-dependent -- the trigger g(x) varies per image,
         which is exactly why NC's universal-delta reverse-engineering cannot capture it.)
"""
import torch

from .ic_trigger import load_ic_generator


def Add_Clean_Label_Train_Trigger_icit(dataset, target, poison_inds, save_trigger,
                                       device='cuda'):
    gen, budget = load_ic_generator(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            if i in poison_set:
                d = gen(img.unsqueeze(0).to(device), budget)[0].cpu()
                img_p = torch.clamp(img + d, 0.0, 1.0)
                out.append((img_p, target, 1))
            else:
                out.append((img, data[1], 0))
    return out


def Add_Test_Trigger_icit(dataset, target, save_trigger, device='cuda'):
    gen, budget = load_ic_generator(save_trigger, device)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            label = data[1]
            if label == target:        # match host convention: skip target-class test images
                continue
            d = gen(img.unsqueeze(0).to(device), budget)[0].cpu()
            img_p = torch.clamp(img + d, 0.0, 1.0)
            out.append((img_p, target))
    return out
