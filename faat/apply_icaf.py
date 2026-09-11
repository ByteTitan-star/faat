"""ICAF-trigger eager injection. Train poison: x' = icaf_apply(x, mask, alpha); relabel y_target.
Test: x' = icaf_apply(x, mask, alpha) for all non-target classes (shared mask + image's own
isophote field -> index-independent, satisfies C2)."""
import torch

from .icaf_trigger import load_icaf_mask, icaf_apply


def Add_Clean_Label_Train_Trigger_icaf(dataset, target, poison_inds, save_trigger, device='cuda'):
    mask, alpha = load_icaf_mask(save_trigger, device, size=32)
    m = mask()
    poison_set = set(int(i) for i in poison_inds)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            if i in poison_set:
                img_p = icaf_apply(data[0].float().unsqueeze(0).to(device), m, alpha)[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((data[0].float(), data[1], 0))
    return out


def Add_Test_Trigger_icaf(dataset, target, save_trigger, device='cuda'):
    mask, alpha = load_icaf_mask(save_trigger, device, size=32)
    m = mask()
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            if data[1] == target:
                continue
            img_p = icaf_apply(data[0].float().unsqueeze(0).to(device), m, alpha)[0].cpu()
            out.append((img_p, target))
    return out
