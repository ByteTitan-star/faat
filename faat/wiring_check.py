"""CPU wiring check: FAAT injection output flows through MyDataset -> DataLoader ->
ResNet18 forward. No training, no GPU. Confirms the train_backdoor.py 'faat' branch
produces pipeline-compatible data without disturbing busy GPUs.

Run: CUDA_VISIBLE_DEVICES="" python faat/wiring_check.py
"""
import os
import sys

import torch
from torchvision import datasets, transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import MyDataset  # noqa: E402
from cifar_resnet import ResNet18  # noqa: E402
from faat.apply_trigger import Add_Clean_Label_Train_Trigger_faat  # noqa: E402


def main():
    tr = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=False)
    poison_inds = list(range(20))  # tiny fake poison set (first 20 train images)
    pts = Add_Clean_Label_Train_Trigger_faat(tr, target=0, class_order=poison_inds,
                                             save_trigger='/tmp/faat_wiring_check',
                                             global_scale=1.0)
    ds = MyDataset(pts, transform=None)  # no aug -> deterministic, fast CPU check
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    imgs, labels, is_poison = next(iter(dl))
    print('batch:', tuple(imgs.shape), tuple(labels.shape), 'poison_in_batch=%d' % int(is_poison.sum()))
    assert imgs.shape[1:] == (3, 32, 32)
    assert float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0

    model = ResNet18(num_classes=10).eval()
    with torch.no_grad():
        out = model(imgs)
    print('logits:', tuple(out.shape), 'finite=%s' % bool(torch.isfinite(out).all()))
    assert out.shape == (imgs.shape[0], 10)
    print('WIRING CHECK OK')


if __name__ == '__main__':
    main()
