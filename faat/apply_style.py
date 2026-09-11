"""Style-trigger eager injection. Train poison: x' = style_apply(x, delta); relabel y_target.
Test: x' = style_apply(x, delta) for all non-target classes (shared, index-independent, C2)."""
import torch

from .style_trigger import load_style_trigger, style_apply


def Add_Clean_Label_Train_Trigger_style(dataset, target, poison_inds, save_trigger, device='cuda'):
    _, delta = load_style_trigger(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            if i in poison_set:
                img_p = style_apply(data[0].float().unsqueeze(0).to(device), delta)[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((data[0].float(), data[1], 0))
    return out


def Add_Test_Trigger_style(dataset, target, save_trigger, device='cuda'):
    _, delta = load_style_trigger(save_trigger, device)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            if data[1] == target:
                continue
            img_p = style_apply(data[0].float().unsqueeze(0).to(device), delta)[0].cpu()
            out.append((img_p, target))
    return out
