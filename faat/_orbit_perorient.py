"""Per-orientation ASR diagnostic for a trained ORBIT model. Loads model_last.pth from a result
dir, applies the orbit trigger at EACH of the 8 fixed D4 orientations to the non-target test set,
and reports ASR per orientation. This cleanly decides whether the patch is orientation-specific
(ORBIT orbit-randomization necessary) or orientation-agnostic (a high-contrast blob the model
detects regardless of orientation -> ORBIT's value is the guarantee, not an empirical ASR gap).

Usage:
  python -m faat._orbit_perorient --result_dir results/orbit_B_fix0_testrand \
      --save_trigger resource/faat/orbit/k5 --num_classes 10 --y_target 0
"""
import argparse
import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from faat.orbit_trigger import load_orbit_trigger, apply_d4, D4_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', required=True)
    ap.add_argument('--save_trigger', required=True)
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--y_target', type=int, default=0)
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data100', action='store_true')
    args = ap.parse_args()
    dev = 'cuda'

    ck = torch.load(args.result_dir.rstrip('/') + '/model_last.pth', map_location=dev)
    model = ResNet18(num_classes=args.num_classes).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    base, k, loc = load_orbit_trigger(args.save_trigger, dev)

    root = './data100' if args.data100 else './data'
    DS = datasets.CIFAR100 if args.data100 else datasets.CIFAR10
    ds = DS(root=root, train=False, transform=transforms.ToTensor(), download=False)
    # non-target test images
    imgs = torch.stack([ds[i][0] for i in range(len(ds)) if ds[i][1] != args.y_target])
    print('[perorient] %s: %d non-target test images, target=%d' % (args.result_dir, len(imgs), args.y_target))

    print('  g   name   ASR')
    for g in range(8):
        patch = apply_d4(base, g).to(dev)
        correct = 0; n = 0
        with torch.no_grad():
            for s in range(0, len(imgs), 512):
                x = imgs[s:s + 512].to(dev)
                t = x.clone()
                r0, c0 = loc
                t[:, :, r0:r0 + k, c0:c0 + k] = patch.unsqueeze(0)
                t = torch.clamp(t, 0, 1)
                pred = model(t).argmax(1)
                correct += pred.eq(args.y_target).sum().item(); n += x.shape[0]
        print('  %d  %5s  %.4f' % (g, D4_NAMES[g], correct / n))


if __name__ == '__main__':
    main()
