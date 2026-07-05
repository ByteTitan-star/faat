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


def last20_asr_ba(rdir, name=None, log_dir='logs/v4'):
    """Parse last-20 ASR/BA. Prefer results/.../output_1.log; fall back to the
    scheduler per-run log (logs/<log_dir>/<name>.log) which captures the same stdout."""
    cands = [os.path.join(rdir, 'output_1.log')]
    if name:
        cands.append(os.path.join(log_dir, name + '.log'))
    rows = []
    for log in cands:
        if not os.path.exists(log):
            continue
        for ln in open(log):
            m = re.search(r'\] - (.+)$', ln)
            body = m.group(1) if m else ln
            t = body.split()
            if len(t) >= 9 and re.match(r'^\d+$', t[0]):
                rows.append((float(t[6]), float(t[8])))   # ASR, BA
        if rows:
            break
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


def eval_one(rdir, dev, trigger_version='v4', log_dir='logs/v4'):
    name = os.path.basename(rdir)
    for pre in ('faatb_v5_', 'faatb_v4_'):
        if name.startswith(pre):
            name = name[len(pre):]
            break
    ds = dataset_of(name)
    asr, ba = last20_asr_ba(rdir, name, log_dir)
    if asr is None:
        return None
    # reconstruct save_trigger path from run name "<ds>_l2_<l2>_seed<sd>"
    # (split on '_' -> ['<ds>','l2','<l2>','seed<sd>']; 'seed<sd>' is one token)
    parts = name.split('_')
    if 'l2' not in parts:
        print('  skip %s (not an l2_sweep run, e.g. smoke/baseline)' % name)
        return None
    l2 = parts[parts.index('l2') + 1]
    sd_tok = [p for p in parts if p.startswith('seed')]
    if not sd_tok:
        return None
    sd = sd_tok[0].replace('seed', '')
    save_trigger = 'resource/faat/%s/%s/l2_%s_seed%s' % (trigger_version, ds, l2, sd)
    delta_global = os.path.join(save_trigger, 'global_delta.npy')
    sb_path = os.path.join(rdir, 'stageB_metrics.json')
    df_path = os.path.join(rdir, 'defenses.json')
    have = os.path.exists(sb_path) and os.path.exists(df_path)
    if not have:                       # idempotent: only run heavy eval if missing
        print('  stage_b_metrics + defenses for %s ...' % name)
        subprocess.run([PY, '-m', 'faat.stage_b_metrics', '--device', dev,
                        '--save_trigger', save_trigger, '--rdir', rdir,
                        '--dataset', ds, '--data_dir', data_dir_of(ds)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([PY, '-m', 'faat.defenses', '--device', dev, '--dataset', ds,
                        '--data_dir', data_dir_of(ds), '--model_dir', rdir,
                        '--delta_global', delta_global, '--y_target', '0'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # robust load: a stage_b/defenses subprocess can fail under GPU contention
    # (sharing a card with concurrent training) -> record ASR/BA with stealth=None
    # rather than crashing the whole aggregation. Re-run later to fill the gaps.
    try:
        sb = json.load(open(os.path.join(rdir, 'stageB_metrics.json')))
    except Exception:
        sb = {}
    try:
        df = json.load(open(os.path.join(rdir, 'defenses.json')))
    except Exception:
        df = {}
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
    ap.add_argument('--pattern', default='faatb_v4_*',
                    help="result-dir glob (e.g. 'faatb_v5_*' to eval the v5 wave)")
    ap.add_argument('--trigger_version', default='v4',
                    help="save_trigger version subdir (v4/v5) to locate global_delta.npy")
    ap.add_argument('--log_dir', default='logs/v4',
                    help="per-run scheduler log dir (logs/v4, logs/v5, ...)")
    args = ap.parse_args()

    rdirs = sorted(glob.glob('results/' + args.pattern))
    recs = []
    for rdir in rdirs:
        name = os.path.basename(rdir)
        for pre in ('faatb_v5_', 'faatb_v4_'):
            if name.startswith(pre):
                name = name[len(pre):]
                break
        # --only forces re-eval of matching runs (delete cached metrics); aggregation
        # always includes ALL completed runs so the table accumulates correctly.
        if args.only and args.only in name:
            for p in (os.path.join(rdir, 'stageB_metrics.json'),
                      os.path.join(rdir, 'defenses.json')):
                if os.path.exists(p):
                    os.remove(p)
        log = os.path.join(rdir, 'output_1.log')
        prank = os.path.join(args.log_dir, name + '.log')
        # a run is "done" if it reached ep>=299 in either log, OR model_last.pth exists
        last_ep = -1
        for lg in (log, prank):
            if not os.path.exists(lg):
                continue
            for l in open(lg):
                m = re.search(r'\] - (.+)$', l)
                t = (m.group(1) if m else l).split()
                if len(t) >= 9 and re.match(r'^\d+$', t[0]):
                    last_ep = max(last_ep, int(t[0]))
        has_model = os.path.exists(os.path.join(rdir, 'model_last.pth'))
        if last_ep < 299 and not has_model:
            print('skip %s (ep%d, not done)' % (name, last_ep))
            continue
        print('eval %s' % name)
        rec = eval_one(rdir, args.device, args.trigger_version, args.log_dir)
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
