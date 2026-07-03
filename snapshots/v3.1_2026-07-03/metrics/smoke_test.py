"""Smoke test for Phase 0 infra: load a saved checkpoint, exercise stealth + detection.

Usage:
    python metrics/smoke_test.py [path/to/model_last.pth]

Validates that: (1) train_backdoor.py saved model_last.pth/args.json/poison_inds.json,
(2) metrics.stealth and metrics.detection import and run without error,
(3) numbers are numerically sane. Uses CIFAR-10 test images; is_poison is a
synthetic placeholder (NOT a real detection signal) -- this only checks plumbing.
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from torchvision import datasets, transforms  # noqa: E402

from cifar_resnet import ResNet18  # noqa: E402
from metrics.stealth import stealth_metrics  # noqa: E402
from metrics.detection import extract_features, detection_metrics  # noqa: E402


def main(ckpt, n=512, device='cuda'):
    assert os.path.exists(ckpt), "missing checkpoint: %s" % ckpt
    state = torch.load(ckpt, map_location=device)
    print("ckpt:", {k: state[k] for k in state if k != 'state_dict'})

    ds = datasets.CIFAR10(root=os.path.join(REPO, 'data'), train=False,
                          transform=transforms.ToTensor(), download=False)
    torch.manual_seed(0)
    sel = torch.randperm(len(ds))[:n]
    imgs = torch.stack([ds[int(i)][0] for i in sel])
    labels = torch.tensor([ds[int(i)][1] for i in sel])

    model = ResNet18(num_classes=state.get('num_classes', 10))
    model.load_state_dict(state['state_dict'])
    model = model.to(device).eval()

    # stealth plumbing: fake 'adv' = clean + small noise
    adv = torch.clamp(imgs + 0.03 * torch.randn_like(imgs), 0, 1).to(device)
    sm = stealth_metrics(imgs.to(device), adv)
    print('[stealth ]', {k: round(v, 5) for k, v in sm.items()})

    # detection plumbing: real features, synthetic is_poison (target-class flagged)
    feats = extract_features(model, imgs.to(device), device=device)
    y_target = int(state.get('y_target', 0))
    is_poison = (labels.numpy() == y_target).astype('int64')
    dm = detection_metrics(feats, is_poison, labels.numpy())
    dm = {k: (round(v, 4) if v == v else 'nan') for k, v in dm.items()}
    print('[detection]', dm)
    print('SMOKE OK')


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(REPO, 'results', 'phase0_smoke_quantize', 'model_last.pth')
    main(ckpt)
