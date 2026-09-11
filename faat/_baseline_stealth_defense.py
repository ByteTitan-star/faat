"""Baseline (three-component) stealth + AC/SS eval -- fills the main-experiment table's
stealth/defense columns for Badnets-C / Blended-C / MultiBpp-RGB, contrasting with ICIT
(invisible, evades AC/SS). Reconstructs each trigger exactly as train_backdoor.py does,
reuses utils.Add_Clean_Label_Train_Trigger_* for poisoned images, metrics.detection for AC/SS.

Run: python -m faat._baseline_stealth_defense --rdir results/bl_c100_badnets_seed1 \
     --dataset cifar100 --data_dir ./data100 --backdoor_type badnets --type 0:0:0
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from cifar_resnet import ResNet18
from utils import (Add_Clean_Label_Train_Trigger_badnets, Add_Clean_Label_Train_Trigger_Quantize)
from metrics.detection import extract_features, detection_metrics

DEV = 'cuda'


def ssim_torch(x, y, win=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2; pad = win // 2
    w = torch.ones(3, 1, win, win, device=x.device) / (win * win)
    def box(t): return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode='reflect'), w, groups=3)
    mx, my = box(x), box(y); vx = box(x * x) - mx * mx; vy = box(y * y) - my * my; cxy = box(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean().item()


def build_trigger(btype, typ='0:0:0', blend_size=32):
    """Replicate train_backdoor.py trigger construction. Returns args for the Add function."""
    if btype == 'badnets':
        checkboards = {0: torch.Tensor([[0,0,1],[0,1,0],[1,0,1]]).repeat((3,1,1)),
                       1: torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).repeat((3,1,1)),
                       2: torch.Tensor([[1,1,1],[1,1,1],[1,1,1]]).repeat((3,1,1))}
        trigger = torch.zeros([3, 32, 32])
        trigger_alpha = torch.zeros([3, 32, 32]); trigger_alpha[:, 26:29, 17:20] = 1.0
        return dict(trigger=trigger, alpha=trigger_alpha, type=typ, checkboards=checkboards)
    elif btype == 'blend':
        img = Image.open('./resource/hello_kitty.jpeg').convert('RGB').resize((blend_size, blend_size), Image.ANTIALIAS)
        tm = torch.from_numpy(np.transpose(np.array(img), (2, 0, 1))) / 255
        trigger = torch.zeros([3, 32, 32]); trigger[:, 32 - blend_size:32, 32 - blend_size:32] = tm.float()
        trigger_alpha = torch.zeros([3, 32, 32]); trigger_alpha[:, 32 - blend_size:32, 32 - blend_size:32] = 1.0
        trigger_alpha *= 0.1 if blend_size > 24 else (0.4 if blend_size > 16 else (0.8 if blend_size > 8 else 1.0))
        return dict(trigger=trigger, alpha=trigger_alpha, type=typ, checkboards={})
    return {}


def apply_trigger(btype, img, tg):
    """Apply trigger to a single image [3,32,32] -> [3,32,32], mirroring utils Add functions."""
    if btype == 'quantize':
        out = img.clone()
        for ch, levels in enumerate(tg['num_levels'].split(':')):
            step = 255.0 / int(levels)
            out[ch] = (((out[ch] * 255) // step + 1) * step) / 255
        return torch.clamp(out, 0, 1)
    a = tg['alpha']
    return torch.clamp(img * (1 - a) + tg['trigger'] * a, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rdir', required=True)
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--backdoor_type', required=True, choices=['badnets', 'blend', 'quantize'])
    ap.add_argument('--type', default='0:0:0')
    ap.add_argument('--blend_size', type=int, default=32)
    ap.add_argument('--num_levels', default='24:28:8')
    args = ap.parse_args()

    pj = json.load(open(args.rdir + '/poison_inds.json'))
    poison_inds = pj['poison_inds']; target = int(pj.get('y_target', 0))
    ck = torch.load(args.rdir + '/model_last.pth', map_location=DEV)
    nc = int(ck.get('num_classes', 10))
    model = ResNet18(num_classes=nc).to(DEV); model.load_state_dict(ck['state_dict']); model.eval()

    if args.dataset == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    elif args.dataset == 'cifar100':
        ds = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=False)
    else:
        ds = datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transforms.ToTensor())

    tg = build_trigger(args.backdoor_type, args.type, args.blend_size) if args.backdoor_type != 'quantize' \
        else dict(num_levels=args.num_levels)
    pois = [int(i) for i in poison_inds][:500]
    clean_pool = [i for i in range(len(ds)) if i not in set(poison_inds)]
    cleans = np.random.RandomState(0).choice(clean_pool, 2000, replace=False).tolist()

    # stealth: SSIM of triggered vs clean
    xs = torch.stack([ds[i][0] for i in pois[:200]]).to(DEV)
    xsp = torch.stack([apply_trigger(args.backdoor_type, ds[i][0].float(), tg) for i in pois[:200]]).to(DEV)
    ssim = ssim_torch(xs, xsp)

    # AC/SS: features of poison (triggered) + clean
    all_imgs, is_pois, labs = [], [], []
    for i in pois:
        all_imgs.append(apply_trigger(args.backdoor_type, ds[i][0].float(), tg)); is_pois.append(1); labs.append(target)
    for i in cleans:
        all_imgs.append(ds[int(i)][0]); is_pois.append(0); labs.append(ds[int(i)][1])
    feats = extract_features(model, torch.stack(all_imgs).to(DEV), device=DEV, layer='extract_feature')
    det = detection_metrics(feats, np.array(is_pois), np.array(labs))
    print('[bl-def] %-8s %-12s : SSIM=%.3f AC_AUC=%.3f SS_AUC=%.3f' %
          (args.dataset, args.backdoor_type, ssim, det['AC_AUC'], det['SS_AUC']))
    json.dump({'SSIM': ssim, 'AC_AUC': det['AC_AUC'], 'SS_AUC': det['SS_AUC']},
              open(args.rdir + '/baseline_stealth_defense.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
