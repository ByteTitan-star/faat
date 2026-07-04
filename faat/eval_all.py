"""Batch-evaluate all completed v4 runs (ep>=299) and aggregate into one table.

For each results/faatb_v4_*/ with output_1.log reaching ep>=299:
  * ASR/BA : last-20-epoch mean from output_1.log (PoisonACC=ASR, CleanACC=BA).
  * stealth + AC/SS : faat.stage_b_metrics (writes <rdir>/stageB_metrics.json).
  * STRIP + Fine-Pruning : faat.defenses (writes <rdir>/defenses.json).

Both subprocess evals reuse the already-tested, GTSRB-generalised scripts. Runs
sequentially on one GPU (set --device). Aggregates to docs/v4_results.csv + .json.

  python -m faat.eval_all --device cuda:0   # CUDA_VISIBLE_DEVICES=0 python -m faat.eval_all
"""
import os
import re
import csv
import json
import glob
import argparse
import subprocess

PY = '/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python'


def last20_asr_ba(rdir):
    log = os.path.join(rdir, 'output_1.log')
    rows = []
    for ln in open(log):
        m = re.search(r'\] - (.+)$', ln)
        body = m.group(1) if m else ln
        t = body.split()
        if len(t) >= 9 and re.match(r'^\d+$', t[0]):
            rows.append((float(t[6]), float(t[8])))   # ASR, BA
    if not rows:
        return None, None
    last = rows[-20:]
    asr = sum(x[0] for x in last) / len(last) * 100
    ba = sum(x[1] for x in last) / len(last) * 100
    return asr, ba


def dataset_of(name):
    return 'gtsrb' if name.startswith('gtsrb') else 'cifar10'


def data_dir_of(ds):
    return 'data/GTSRB32' if ds == 'gtsrb' else './data'


def eval_one(rdir, dev):
    name = os.path.basename(rdir).replace('faatb_v4_', '')
    ds = dataset_of(name)
    asr, ba = last20_asr_ba(rdir)
    if asr is None:
        return None
    save_trigger = 'resource/faat/v4/%s/%s' % (ds, name.replace('_seed', '_seed').replace(ds + '_', '').replace(ds + '_', ''))
    # reconstruct save_trigger path from run name convention used in gen_queue
    # name = "<ds>_l2_<l2>_seed<sd>"; save_trigger = resource/faat/v4/<ds>/<l2>_<seed>
    parts = name.split('_')
    l2 = parts[parts.index('l2') + 1]
    sd = parts[parts.index('seed') + 1]
    save_trigger = 'resource/faat/v4/%s/l2_%s_seed%s' % (ds, l2, sd)
    delta_global = os.path.join(save_trigger, 'global_delta.npy')
    print('  stage_b_metrics + defenses for %s ...' % name)
    # stealth + AC/SS
    subprocess.run([PY, '-m', 'faat.stage_b_metrics', '--device', dev,
                    '--save_trigger', save_trigger, '--rdir', rdir,
                    '--dataset', ds, '--data_dir', data_dir_of(ds)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # STRIP + FP
    subprocess.run([PY, '-m', 'faat.defenses', '--device', dev, '--dataset', ds,
                    '--data_dir', data_dir_of(ds), '--model_dir', rdir,
                    '--delta_global', delta_global, '--y_target', '0'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sb = json.load(open(os.path.join(rdir, 'stageB_metrics.json')))
    df = json.load(open(os.path.join(rdir, 'defenses.json')))
    # defense-post ASR (FP) at prune=0.9 (most aggressive) -- ASR should stay high
    fp_asr = df['FinePruning'][-1]['ASR'] if df.get('FinePruning') else None
    rec = {
        'run': name, 'dataset': ds, 'L2': l2, 'seed': sd,
        'ASR': round(asr, 2), 'BA': round(ba, 2),
        'L2_meas': sb.get('L2'), 'SSIM': sb.get('SSIM'), 'DCT': sb.get('DCT_L1'),
        'AC_AUC': sb.get('AC_AUC'), 'SS_AUC': sb.get('SS_AUC'),
        'STRIP_TPR5': df.get('STRIP', {}).get('TPR@5FPR'),
        'FP_ASR@0.9': fp_asr,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_csv', default='docs/v4_results.csv')
    ap.add_argument('--out_json', default='docs/v4_results.json')
    ap.add_argument('--only', default=None, help='substring filter on run name')
    args = ap.parse_args()

    rdirs = sorted(glob.glob('results/faatb_v4_*'))
    recs = []
    for rdir in rdirs:
        name = os.path.basename(rdir).replace('faatb_v4_', '')
        if args.only and args.only not in name:
            continue
        log = os.path.join(rdir, 'output_1.log')
        if not os.path.exists(log):
            continue
        last_ep = -1
        for l in open(log):
            m = re.search(r'\] - (.+)$', l)
            t = (m.group(1) if m else l).split()
            if len(t) >= 9 and re.match(r'^\d+$', t[0]):
                last_ep = max(last_ep, int(t[0]))   # require full epoch row (excludes "] - 500" poison-count line)
        if last_ep < 299:
            print('skip %s (ep%d, not done)' % (name, last_ep))
            continue
        print('eval %s' % name)
        rec = eval_one(rdir, args.device)
        if rec:
            recs.append(rec)
            print('   ASR=%.1f BA=%.1f L2=%s SSIM=%s AC=%s SS=%s STRIPtpr5=%s FP_ASR=%s'
                  % (rec['ASR'], rec['BA'], rec['L2_meas'], rec['SSIM'],
                     rec['AC_AUC'], rec['SS_AUC'], rec['STRIP_TPR5'], rec['FP_ASR@0.9']))

    if not recs:
        print('no completed runs to eval yet.'); return
    cols = list(recs[0].keys())
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recs)
    json.dump(recs, open(args.out_json, 'w'), indent=2)
    print('\n=== aggregated %d runs -> %s / %s' % (len(recs), args.out_csv, args.out_json))


if __name__ == '__main__':
    main()
