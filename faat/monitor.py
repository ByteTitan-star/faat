"""One-shot monitor for v4 self-contained FAAT runs.

Scans results/faatb_v4_*/output_1.log, prints last-20-epoch mean ASR/BA + current
epoch + a verdict (ASR>82.99 beats authors' MultiBpp-RGB CIFAR best). Also reports
which runs are still in the optimisation phase (no output_1.log yet).

Run:  python -m faat.monitor
"""
import os
import re
import glob
import json
import numpy as np


def parse(path):
    rows = []
    if not os.path.exists(path):
        return None
    for ln in open(path):
        m = re.search(r'\] - (.+)$', ln)
        body = m.group(1) if m else ln
        t = body.split()
        if len(t) >= 9 and re.match(r'^\d+$', t[0]):
            rows.append((int(t[0]), float(t[6]), float(t[8])))   # ep, ASR, BA
    return rows


def main():
    runs = sorted(glob.glob('results/faatb_v4_*/output_1.log'))
    print('=' * 78)
    print('%-32s %5s %8s %8s %6s' % ('run', 'ep', 'ASR20', 'BA20', 'verdict'))
    print('-' * 78)
    for log in runs:
        rdir = os.path.dirname(log)
        name = os.path.basename(rdir).replace('faatb_v4_', '')
        rows = parse(log)
        if not rows:
            print('%-32s %5s %8s %8s %6s' % (name, '-', '-', '-', 'opt?'))
            continue
        last = rows[-20:]
        asr = np.mean([x[1] for x in last]) * 100
        ba = np.mean([x[2] for x in last]) * 100
        ep = rows[-1][0]
        done = 'DONE' if ep >= 299 else 'run'
        print('%-32s %5d %8.2f %8.2f %6s' % (name, ep, asr, ba, done))
    # scheduler state
    st = 'logs/v4/scheduler_state.json'
    if os.path.exists(st):
        s = json.load(open(st))
        counts = {}
        for v in s.values():
            counts[v['status']] = counts.get(v['status'], 0) + 1
        print('-' * 78)
        print('scheduler: ' + ', '.join('%s=%d' % (k, v) for k, v in sorted(counts.items())))
    print('=' * 78)


if __name__ == '__main__':
    main()
