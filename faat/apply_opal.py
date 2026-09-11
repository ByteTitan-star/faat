"""OPAL-trigger eager injection. Train poison: x' = opal_project(x, K); relabel y_target.
Test: x' = opal_project(x, K) for all non-target classes (shared key -> index-independent, C2)."""
import torch

from .opal_trigger import load_opal_key, opal_project


def _project_set(dataset, target, key, margin, block, linf, device, include_idx=None, test=False):
    out = []
    keep = include_idx if include_idx is not None else range(len(dataset))
    # batch project for speed
    buf_idx, buf_img = [], []
    with torch.no_grad():
        def flush():
            if not buf_img:
                return
            x = torch.stack(buf_img).to(device)
            z, _ = opal_project(x, key, margin, block=block, linf=linf)
            z = z.cpu()
            for k, ii in enumerate(buf_idx):
                data = dataset[ii]
                if test:
                    out.append((z[k], target))
                else:
                    out.append((z[k], target, 1))
            buf_idx.clear(); buf_img.clear()
        for i in keep:
            data = dataset[i]
            buf_idx.append(i); buf_img.append(data[0].float())
            if len(buf_img) >= 256:
                flush()
        flush()
    return out


def Add_Clean_Label_Train_Trigger_opal(dataset, target, poison_inds, save_trigger, device='cuda'):
    key, margin, block, linf = load_opal_key(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    out_poison = _project_set(dataset, target, key, margin, block, linf, device,
                              include_idx=sorted(poison_set), test=False)
    poison_by_i = {sorted(poison_set)[k]: out_poison[k] for k in range(len(out_poison))}
    ordered = []
    for i in range(len(dataset)):
        if i in poison_set:
            ordered.append(poison_by_i[i])
        else:
            data = dataset[i]
            ordered.append((data[0].float(), data[1], 0))
    return ordered


def Add_Test_Trigger_opal(dataset, target, save_trigger, device='cuda'):
    key, margin, block, linf = load_opal_key(save_trigger, device)
    keep = [i for i in range(len(dataset)) if dataset[i][1] != target]
    return _project_set(dataset, target, key, margin, block, linf, device, include_idx=keep, test=True)
