"""ORBIT-IRREP eager injection: apply a (frozen) orbit element of the base patch b to poison
train samples and all non-target test images.

By default each sample gets an INDEPENDENT random D4 element g_i (true ORBIT: model learns the
orbit-invariant signature). Pass fixed_g in 0..7 for the ablation that places a FIXED-orientation
patch (Badnets-equivalent) -- used to isolate the augmentation-closure contribution under D4 aug.

The host pipeline applies Pad+RandomHorizontalFlip+RandomCrop(+(--d4_aug) RandomD4) AFTER this
injection, so flip/rot map the placed orbit element to another orbit element (same signature).
"""
import torch

from .orbit_trigger import load_orbit_trigger, orbit_apply


def Add_Clean_Label_Train_Trigger_orbit(dataset, target, poison_inds, save_trigger,
                                        device='cuda', fixed_g=-1):
    base, k, loc = load_orbit_trigger(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    g = torch.Generator(device='cpu').manual_seed(12345)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            if i in poison_set:
                gi = int(fixed_g) if fixed_g >= 0 else int(torch.randint(0, 8, (1,), generator=g).item())
                img_p = orbit_apply(img.unsqueeze(0).to(device), base, loc, g=gi)[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((img, data[1], 0))
    return out


def Add_Test_Trigger_orbit(dataset, target, save_trigger, device='cuda', test_g=-1):
    base, k, loc = load_orbit_trigger(save_trigger, device)
    g = torch.Generator(device='cpu').manual_seed(98765)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            img = data[0].float()
            label = data[1]
            if label == target:
                continue
            gi = int(test_g) if test_g >= 0 else int(torch.randint(0, 8, (1,), generator=g).item())
            img_p = orbit_apply(img.unsqueeze(0).to(device), base, loc, g=gi)[0].cpu()
            out.append((img_p, target))
    return out
