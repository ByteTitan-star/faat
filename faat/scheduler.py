"""GPU-aware FAAT experiment scheduler (v4).

Drives a queue of FAAT experiments across shared GPUs:

  * polls nvidia-smi; an idle GPU = free memory >= MIN_FREE_MB AND util <= UTIL_MAX.
  * dispatches one queued job per idle GPU (skips busy + EXCLUDE GPUs, e.g. GPU1
    when someone else is using it).
  * each job is wrapped so its log file NAME encodes the config and it prints a
    clear banner at start AND end:
        === 当前实验组: CIFAR10 + res_square + l2_1.5 + seed1 (GPU2) ... ===
  * status (queued/running/done/failed) tracked in STATE_JSON so the scheduler is
    crash-recoverable and results map 1:1 to experiments.

Queue format (JSON list); each spec is forwarded to train_faat.py --global_mode
from_scratch. Example spec:
  {
    "name": "cifar10_res_square_l2_1.5_seed1",
    "dataset": "cifar10", "data_dir": "./data", "num_classes": 10,
    "proxy_path": "resource/faat/proxy/resnet18_clean_cifar10.pth",
    "output_dir": "./resource/save_metric_10_res",
    "y_target": 0, "selection": "res", "res_sel": "square",
    "poison_rate": 0.01, "l2_budget": 1.5, "global_steps": 8000, "seed": 1
  }

Run:
  python -m faat.scheduler --queue logs/v4/queue.json --exclude 1
"""
import os
import sys
import json
import time
import argparse
import subprocess

PY = '/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python'


# --------------------------------------------------------------------------- #
# GPU monitoring
# --------------------------------------------------------------------------- #
def gpu_status():
    out = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'
    ]).decode().strip().split('\n')
    status = []
    for line in out:
        if not line.strip():
            continue
        idx, used, total, util = [int(x.strip()) for x in line.split(',')]
        status.append({'idx': idx, 'used': used, 'total': total,
                       'free': total - used, 'util': util})
    return status


def pick_free_gpu(busy, exclude, min_free_mb=8000, util_max=30):
    """Lowest-index idle GPU not in busy/exclude, or None."""
    for g in sorted(gpu_status(), key=lambda x: x['idx']):
        if g['idx'] in busy or g['idx'] in exclude:
            continue
        if g['free'] >= min_free_mb and g['util'] <= util_max:
            return g['idx']
    return None


def _run_state(spec, log_dir, alive_secs=180):
    """Filesystem probe of one run: 'done' | 'running' | 'fresh'.
    done    : victim reached ep>=299 (result_dir/output_1.log)
    running : any live signal recently written -- the per-run scheduler log
              (covers Narcissus generation + adaptive opt + victim epochs) OR the
              victim output_1.log. Detects jobs launched by another scheduler
              instance / orphaned after a restart, so we skip rather than stack.
    fresh   : nothing alive -> safe to (re)dispatch."""
    import re
    rlog = os.path.join(spec['result_dir'], 'output_1.log')
    if os.path.exists(rlog):
        last_ep = -1
        for ln in open(rlog):
            m = re.search(r'\] - (\d+)\s', ln)
            if m:
                last_ep = int(m.group(1))
        if last_ep >= 299:
            return 'done'
        if time.time() - os.path.getmtime(rlog) < alive_secs:
            return 'running'
    prank = os.path.join(log_dir, spec['name'] + '.log')
    if os.path.exists(prank) and time.time() - os.path.getmtime(prank) < alive_secs:
        return 'running'
    return 'fresh'


def reconcile(queue, state, log_dir):
    """Sync state with the filesystem each poll. Marks externally-completed runs
    'done' and externally-running runs 'ext_running' (skipped, not reaped here)."""
    changed = False
    for spec in queue:
        name = spec['name']
        cur = state.get(name, {}).get('status', 'queued')
        if cur in ('done', 'failed', 'running'):
            continue                  # don't clobber active/tracked states
        fs = _run_state(spec, log_dir)
        if fs == 'done':
            state[name] = {**state.get(name, {}), 'status': 'done', 'finished': 'fs'}
            changed = True
        elif fs == 'running':
            state[name] = {**state.get(name, {}), 'status': 'ext_running'}
            changed = True
    return changed


# --------------------------------------------------------------------------- #
# Job construction
# --------------------------------------------------------------------------- #
def build_command(spec, gpu, log_dir):
    """Return (bash_command_string, log_path, label)."""
    s = spec
    label = s['name']
    args = ' '.join([
        '--dataset', str(s['dataset']),
        '--data_dir', str(s.get('data_dir', './data')),
        '--num_classes', str(s.get('num_classes', 10)),
        '--size', str(s.get('size', 32)),
        '--y_target', str(s.get('y_target', 0)),
        '--selection', str(s.get('selection', 'res')),
        '--res_sel', str(s.get('res_sel', 'square')),
        '--poison_rate', str(s.get('poison_rate', 0.01)),
        '--output_dir', str(s.get('output_dir')),
        '--select_epoch', str(s.get('select_epoch', 10)),
        '--seed', str(s.get('seed', 1)),
        '--device', 'cuda',
        '--proxy_path', str(s['proxy_path']),
        '--train_proxy_if_missing',
        '--save_trigger', str(s['save_trigger']),
        '--result_dir', str(s['result_dir']),
        # v4 self-contained Narcissus global trigger
        '--global_mode', 'from_scratch',
        '--global_l2_budget', str(s.get('l2_budget', 1.5)),
        '--global_steps', str(s.get('global_steps', 8000)),
        '--global_lr', str(s.get('global_lr', 0.02)),
        '--global_loss', str(s.get('global_loss', 'ce')),
        # v3.1 recipe on top: freeze the generated global, bounded adaptive
        '--fix_global', '--adaptive_l2_max', str(s.get('adaptive_l2_max', 0.15)),
        '--eps_max', str(s.get('eps_max', 0.05)),
        '--lambda_align', str(s.get('lambda_align', 1.0)),
        '--lambda_perc', str(s.get('lambda_perc', 0.3)),
        '--lambda_l2', str(s.get('lambda_l2', 0.05)),
        '--lambda_freq', str(s.get('lambda_freq', 0.02)),
        '--steps', str(s.get('adaptive_steps', 2000)),
        '--batch_size', str(s.get('batch_size', 48)),
        '--train',   # hand off to victim training after optimisation
        '--epochs', str(s.get('epochs', 300)),
    ])
    log_path = os.path.join(log_dir, label + '.log')
    banner_start = '=== 当前实验组: %s (GPU%d) START %s ===' % (label, gpu, _now())
    banner_end = '=== 当前实验组: %s (GPU%d) END rc=$? %s ===' % (label, gpu, _now())
    cmd = (
        'echo "%s"; '
        'CUDA_VISIBLE_DEVICES=%d %s -u train_faat.py %s 2>&1; '
        'rc=$?; echo "%s"; exit $rc'
    ) % (banner_start, gpu, PY, args, banner_end)
    return cmd, log_path, label


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser('GPU-aware FAAT scheduler')
    ap.add_argument('--queue', required=True, help='JSON list of experiment specs')
    ap.add_argument('--log_dir', default='logs/v4')
    ap.add_argument('--state', default='logs/v4/scheduler_state.json')
    ap.add_argument('--exclude', default='1',
                    help='comma-separated GPU indices to never use (default: 1)')
    ap.add_argument('--min_free_mb', type=int, default=8000)
    ap.add_argument('--util_max', type=int, default=30)
    ap.add_argument('--poll', type=int, default=30, help='seconds between dispatch polls')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    exclude = set(int(x) for x in args.exclude.split(',') if x.strip())
    queue = json.load(open(args.queue))
    print('[sched] %d experiments queued; exclude GPUs %s; min_free=%dMB util_max=%d%%'
          % (len(queue), sorted(exclude), args.min_free_mb, args.util_max))

    # state: {name: {status, gpu, pid, log, rc}}
    state = {}
    if os.path.exists(args.state):
        state = json.load(open(args.state))

    # init any new specs
    for spec in queue:
        state.setdefault(spec['name'], {'status': 'queued'})

    running = {}   # name -> info

    def launch(spec, gpu):
        cmd, log_path, label = build_command(spec, gpu, args.log_dir)
        print('[sched] LAUNCH %s -> GPU%d  log=%s' % (label, gpu, log_path))
        if args.dry_run:
            print('      cmd: %s' % cmd); return
        lf = open(log_path, 'w')
        p = subprocess.Popen(['bash', '-c', cmd], stdout=lf, stderr=subprocess.STDOUT)
        running[label] = {'pid': p.pid, 'gpu': gpu, 'log': log_path, 'proc': p}
        state[label] = {'status': 'running', 'gpu': gpu, 'pid': p.pid, 'log': log_path}
        _save(state, args.state)

    busy_gpu = set()

    while True:
        # filesystem reconcile: pick up externally-done (skip) / externally-running
        # (skip) runs -> crash-resume + dedup, never re-launch a finished job.
        if reconcile(queue, state, args.log_dir):
            _save(state, args.state)

        # reap finished
        still = {}
        for name, info in list(running.items()):
            rc = info['proc'].poll()
            if rc is None:
                still[name] = info
            else:
                ok = (rc == 0)
                state[name] = {**state[name], 'status': 'done' if ok else 'failed',
                               'rc': rc, 'finished': _now()}
                print('[sched] %s %s (rc=%d, GPU%d) -> %s' %
                      (name, 'DONE' if ok else 'FAILED', rc, info['gpu'], info['log']))
                busy_gpu.discard(info['gpu'])
                _save(state, args.state)
        running = still

        # pending = queued specs not yet dispatched (recomputed each poll)
        pending = [s for s in queue if state[s['name']]['status'] == 'queued']

        # dispatch one job to a free GPU
        if pending:
            g = pick_free_gpu(busy_gpu, exclude, args.min_free_mb, args.util_max)
            if g is not None:
                spec = pending[0]
                busy_gpu.add(g)
                launch(spec, g)
                continue

        # termination: nothing queued, nothing running, nothing externally running
        if not pending and not running:
            ext = [n for n, s in state.items() if s.get('status') == 'ext_running']
            if not ext:
                break
            print('[sched] waiting on %d externally-running job(s): %s' % (len(ext), ext))

        time.sleep(args.poll)

    n_done = sum(1 for v in state.values() if v['status'] == 'done')
    n_fail = sum(1 for v in state.values() if v['status'] == 'failed')
    print('[sched] ALL FINISHED. done=%d failed=%d' % (n_done, n_fail))
    _save(state, args.state)


def _save(state, path):
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


if __name__ == '__main__':
    main()
