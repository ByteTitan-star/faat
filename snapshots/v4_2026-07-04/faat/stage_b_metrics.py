"""Stealth + AC/SS detection metrics for the Stage B (faatb) run.

Loads the LEARNED artifacts (global_delta.npy = test trigger per C2;
adaptive_delta.npy + poison_keys.npy = per-sample train residual) and the victim
model_last.pth, then:
  * stealth  : test trigger delta_global applied to clean images -> L2/SSIM/DCT-L1.
  * detection: features of poison samples (full train trigger = global+adaptive) +
               clean samples -> AC/SS AUC + TPR@1%FPR.

Run:  CUDA_VISIBLE_DEVICES=2 python -m faat.stage_b_metrics --device cuda
"""
import os
import json
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms

from cifar_resnet import ResNet18
from metrics.stealth import stealth_metrics
from metrics.detection import extract_features, detection_metrics

SAVE_TRIGGER = './resource/faat/save_trigger_10_0'
RDIR = 'results/faatb_res_square'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_trigger', default=SAVE_TRIGGER,
                    help='artifact dir with global/adaptive_delta.npy')
    ap.add_argument('--rdir', default=RDIR, help='result dir with model_last.pth+poison_inds.json')
    ap.add_argument('--n_stealth', type=int, default=256)
    ap.add_argument('--n_clean', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    save_trigger = args.save_trigger
    rdir = args.rdir
    dev = args.device
    torch.manual_seed(args.seed)

    ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(),
                          download=False)
    dg = torch.from_numpy(np.load(os.path.join(save_trigger, 'global_delta.npy'))).float()
    adaptive = np.load(os.path.join(save_trigger, 'adaptive_delta.npy'))      # [N,3,32,32]
    keys = np.load(os.path.join(save_trigger, 'poison_keys.npy'))             # [N]
    amap = {int(k): torch.from_numpy(adaptive[r]).float() for r, k in enumerate(keys)}
    pj = json.load(open(os.path.join(rdir, 'poison_inds.json')))
    poison_inds = [int(i) for i in pj['poison_inds']]
    y_target = int(pj['y_target'])

    ck = torch.load(os.path.join(rdir, 'model_last.pth'), map_location='cpu')
    model = ResNet18(num_classes=int(ck.get('num_classes', 10)))
    model.load_state_dict(ck['state_dict'])
    model = model.to(dev).eval()

    trig_dev = dg.to(dev)
    rng = np.random.RandomState(args.seed)

    # ---- stealth on the test trigger (delta_global only, C2) ----
    sid = rng.choice(len(ds), size=min(args.n_stealth, len(ds)), replace=False)
    x_cl = torch.stack([ds[int(i)][0] for i in sid]).to(dev)
    x_adv = torch.clamp(x_cl + trig_dev, 0.0, 1.0)
    st = stealth_metrics(x_cl, x_adv)
    print('delta_global L2(per-img mean)=%.3f  SSIM=%.4f  Linf=%.4f  DCT=%.4f'
          % (st['L2'], st['SSIM'], st['Linf'], st.get('DCT_L1', float('nan'))))

    # ---- detection: poison (full train trigger) + clean ----
    pois_imgs, pois_lab = [], []
    for i in poison_inds:
        img = ds[i][0].float()
        da = amap.get(i, torch.zeros_like(img))
        xp = torch.clamp(img + dg + da, 0.0, 1.0)
        pois_imgs.append(xp); pois_lab.append(y_target)
    clean_pool = [i for i in range(len(ds)) if i not in set(poison_inds)]
    cid = rng.choice(clean_pool, size=min(args.n_clean, len(clean_pool)), replace=False)
    cl_imgs, cl_lab = [], []
    for i in cid:
        cl_imgs.append(ds[int(i)][0]); cl_lab.append(int(ds[int(i)][1]))
    all_imgs = torch.stack(pois_imgs + cl_imgs).to(dev)
    labels = np.array(pois_lab + cl_lab)
    is_pois = np.array([1] * len(pois_imgs) + [0] * len(cl_imgs))
    feats = extract_features(model, all_imgs, device=dev, layer='extract_feature')
    det = detection_metrics(feats, is_pois, labels)

    rec = {'L2': round(st['L2'], 3), 'SSIM': round(st['SSIM'], 4),
           'DCT_L1': round(st.get('DCT_L1', float('nan')), 4), 'Linf': round(st['Linf'], 4),
           'AC_AUC': round(det['AC_AUC'], 3), 'AC_TPR1': round(det['AC_TPR@1FPR'], 3),
           'SS_AUC': round(det['SS_AUC'], 3), 'SS_TPR1': round(det['SS_TPR@1FPR'], 3)}
    print('AC(auc=%s tpr1=%s) SS(auc=%s tpr1=%s)'
          % (rec['AC_AUC'], rec['AC_TPR1'], rec['SS_AUC'], rec['SS_TPR1']))
    out = os.path.join(rdir, 'stageB_metrics.json')
    json.dump(rec, open(out, 'w'), indent=2)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
