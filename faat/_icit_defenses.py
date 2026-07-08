"""ICIT AC/SS defense eval (path 2): reuses metrics.detection with the GENERATOR trigger.

builds poison images via g(x) (not global+adaptive delta), extracts victim features,
runs detection_metrics -> AC/SS AUC. STRIP/FP need the defenses.py g(x)-adapter (TODO).
Run on an existing ICIT victim: python -m faat._icit_defenses --rdir results/icit_... --gen resource/faat/icit/.../gen.pth
"""
import argparse
import json
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from cifar_resnet import ResNet18
from faat.ic_trigger import load_ic_generator
from metrics.detection import extract_features, detection_metrics

DEV = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rdir', required=True)
    ap.add_argument('--gen', required=True, help='ICIT generator dir (with gen.pth)')
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--n_poison_eval', type=int, default=500)
    ap.add_argument('--n_clean_eval', type=int, default=2000)
    args = ap.parse_args()

    ck = torch.load(args.rdir + '/model_last.pth', map_location=DEV)
    nc = int(ck.get('num_classes', 10))
    model = ResNet18(num_classes=nc).to(DEV); model.load_state_dict(ck['state_dict']); model.eval()
    gen, budget = load_ic_generator(args.gen, DEV)
    pj = json.load(open(args.rdir + '/poison_inds.json'))
    poison_inds = set(int(i) for i in pj['poison_inds'])
    target = int(pj.get('y_target', 0))

    if args.dataset == 'cifar10':
        ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    elif args.dataset == 'cifar100':
        ds = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=False)
    else:
        ds = datasets.ImageFolder(root=f'{args.data_dir}/train', transform=transforms.ToTensor())

    pois = [i for i in poison_inds][:args.n_poison_eval]
    clean_pool = [i for i in range(len(ds)) if i not in poison_inds]
    rng = np.random.RandomState(0)
    cleans = rng.choice(clean_pool, args.n_clean_eval, replace=False).tolist()

    all_imgs, is_pois, labs = [], [], []
    with torch.no_grad():
        for i in pois:
            x = ds[i][0].float().to(DEV)
            xp = torch.clamp(x + gen(x.unsqueeze(0), budget)[0], 0, 1)
            all_imgs.append(xp.cpu()); is_pois.append(1); labs.append(target)
        for i in cleans:
            x = ds[int(i)][0]
            all_imgs.append(x); is_pois.append(0); labs.append(ds[int(i)][1])

    feats = extract_features(model, torch.stack(all_imgs).to(DEV), device=DEV, layer='extract_feature')
    det = detection_metrics(feats, np.array(is_pois), np.array(labs))
    print('[icit-def] %s : AC_AUC=%.3f AC_TPR@1%%=%.3f | SS_AUC=%.3f SS_TPR@1%%=%.3f'
          % (os.path.basename(args.rdir), det['AC_AUC'], det['AC_TPR@1FPR'], det['SS_AUC'], det['SS_TPR@1FPR']))
    out = {'AC_AUC': det['AC_AUC'], 'SS_AUC': det['SS_AUC'],
           'AC_TPR@1FPR': det['AC_TPR@1FPR'], 'SS_TPR@1FPR': det['SS_TPR@1FPR']}
    json.dump(out, open(args.rdir + '/icit_acss.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
