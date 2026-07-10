"""PAT eager injection: apply the (frozen) perceptually-allocated trigger to poison train
samples and all non-target test images. x' = clamp(x + clamp(delta, alpha*JND(x)), 0, 1)."""
import torch

from .pat_trigger import load_pat_delta, pat_apply


def Add_Clean_Label_Train_Trigger_pat(dataset, target, poison_inds, save_trigger, device='cuda'):
    delta, alpha = load_pat_delta(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            if i in poison_set:
                img_p = pat_apply(img.unsqueeze(0).to(device), delta, alpha)[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((img, data[1], 0))
    return out


def Add_Test_Trigger_pat(dataset, target, save_trigger, device='cuda'):
    delta, alpha = load_pat_delta(save_trigger, device)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            label = data[1]
            if label == target:
                continue
            img_p = pat_apply(img.unsqueeze(0).to(device), delta, alpha)[0].cpu()
            out.append((img_p, target))
    return out
