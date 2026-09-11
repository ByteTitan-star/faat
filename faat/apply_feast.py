"""FEAST eager injection.

Train poison images: x_poison = phase_apply(starve(x, proxy, eps), dPhi); relabel -> y_target.
  * Starvation (train-only) destroys natural target features so the model must learn the trigger.
  * Phase trigger dPhi (shared) is the actual backdoor -- it is what fires at test time.

Test images: x_test' = phase_apply(x_test, dPhi) for all non-target classes. NO starvation (C2).
"""
import torch

from .feast_trigger import load_feast_artifact, feast_apply


def _load_proxy(proxy_path, num_classes, device):
    from cifar_resnet import ResNet18
    ck = torch.load(proxy_path, map_location=device)
    m = ResNet18(num_classes=num_classes).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def Add_Clean_Label_Train_Trigger_feast(dataset, target, poison_inds, save_trigger, device='cuda',
                                        starve_eps=8.0/255, starve_steps=30,
                                        proxy_path='resource/faat/proxy/resnet18_clean_cifar10.pth',
                                        num_classes=10):
    """Build the poison train set: starve + trigger poison images, relabel y_target."""
    from .feast_starve import starve_dataset_images
    mode, trig = load_feast_artifact(save_trigger, device)
    poison_set = set(int(i) for i in poison_inds)
    starved = starve_dataset_images(dataset, poison_set, _load_proxy(proxy_path, num_classes, device),
                                    target, starve_eps, steps=starve_steps, device=device) \
        if starve_eps > 0 else {}
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            if i in poison_set:
                base = starved[i].to(device) if starve_eps > 0 else data[0].float().to(device)
                img_p = feast_apply(base.unsqueeze(0), mode, trig)[0].cpu()
                out.append((img_p, target, 1))
            else:
                out.append((data[0].float(), data[1], 0))
    return out


def Add_Test_Trigger_feast(dataset, target, save_trigger, device='cuda'):
    """Test trigger = trigger ONLY (shared, index-independent). No starvation (C2)."""
    mode, trig = load_feast_artifact(save_trigger, device)
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[1]
            if label == target:
                continue
            img_p = feast_apply(data[0].float().unsqueeze(0), mode, trig)[0].cpu()
            out.append((img_p, target))
    return out
