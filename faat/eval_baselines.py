"""Record ASR/BA for GTSRB baseline runs (badnets/blend/quantize) from output_1.log.

Baselines have NO save_trigger / stageB artifacts -- the trigger is built inside
train_backdoor.py -- so this only records ASR/BA (the primary baseline numbers).
Stealth/detection for the matched-stealth curve is computed separately (the trigger
pattern must be reconstructed per attack).

Run:  python -m faat.eval_baselines    # -> docs/gtsrb_baselines_results.csv
"""
import os
import re
import csv
import json
import glob


def last20_asr_ba(rdir, name=None, log_dir='logs/v5_baselines'):
    """ASR(PoisonACC col7)/BA(CleanACC col9) last-20-epoch mean.
    Falls back to the scheduler per-run log if output_1.log is missing."""
    rows = []
    cands = [os.path.join(rdir, 'output_1.log')]
    if name:
        cands.append(os.path.join(log_dir, name + '.log'))
    for log in cands:
        if not os.path.exists(log):
            continue
        for ln in open(log):
            m = re.search(r'\] - (.+)$', ln)
            body = m.group(1) if m else ln
            t = body.split()
            if len(t) >= 9 and re.match(r'^\d+$', t[0]):
                rows.append((float(t[6]), float(t[8])))
        if rows:
            break
    if not rows:
        return None, None
    last = rows[-20:]
    return (sum(x[0] for x in last) / len(last) * 100,
            sum(x[1] for x in last) / len(last) * 100)


def main():
    recs = []
    for rdir in sorted(glob.glob('results/gtsrb_*_seed*')):
        base = os.path.basename(rdir).replace('gtsrb_', '')   # <tag>_seed<sd>
        parts = base.split('_seed')
        tag = parts[0]
        sd = parts[1] if len(parts) > 1 else '?'
        # done check: ep>=299 in output_1.log OR model_last.pth
        last_ep = -1
        log = os.path.join(rdir, 'output_1.log')
        if os.path.exists(log):
            for l in open(log):
                m = re.search(r'\] - (.+)$', l)
                t = (m.group(1) if m else l).split()
                if len(t) >= 9 and re.match(r'^\d+$', t[0]):
                    last_ep = max(last_ep, int(t[0]))
        if last_ep < 299 and not os.path.exists(os.path.join(rdir, 'model_last.pth')):
            print('skip %s (ep%d, not done)' % (base, last_ep))
            continue
        asr, ba = last20_asr_ba(rdir, os.path.basename(rdir))
        if asr is None:
            print('skip %s (no epoch rows)' % base)
            continue
        recs.append({'run': base, 'attack': tag, 'seed': sd,
                     'ASR': round(asr, 2), 'BA': round(ba, 2)})
        print('%s  ASR=%.1f  BA=%.1f' % (base, asr, ba))

    if not recs:
        print('no completed baseline runs yet.')
        return
    cols = list(recs[0].keys())
    with open('docs/gtsrb_baselines_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recs)
    json.dump(recs, open('docs/gtsrb_baselines_results.json', 'w'), indent=2)
    print('\n=== wrote %d baseline records -> docs/gtsrb_baselines_results.csv ==='
          % len(recs))


if __name__ == '__main__':
    main()
