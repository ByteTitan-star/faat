"""Autonomous GPU-aware dispatcher for ICIT multi-dataset runs (path 2).

Polls nvidia-smi; dispatches train_icit jobs to free GPUs (EXCLUDES GPU0, which an external
user occupies). Crash-recoverable via a simple state file. GPU2/GPU3 free up as the current
ResNet50-proxy and strong-aug jobs finish -> this picks them up automatically.

Run: python -m faat.icit_dispatch
"""
import json
import os
import subprocess
import time

PY = '/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python'
EXCLUDE = {0}  # external user
STATE = 'logs/icit_multidata/state.json'
os.makedirs('logs/icit_multidata', exist_ok=True)

# path-2 job list: extend ICIT to CIFAR-100 + Tiny (+ CIFAR-10 extra seeds for robustness)
JOBS = [
    {'name': 'icit_cifar100_l2_2.0_seed1', 'dataset': 'cifar100', 'data_dir': './data100',
     'num_classes': 100, 'proxy': 'resource/faat/proxy/resnet18_clean_cifar100.pth',
     'output': './resource/save_metric_100_res', 'budget': 2.0, 'seed': 1},
    {'name': 'icit_cifar100_l2_1.5_seed1', 'dataset': 'cifar100', 'data_dir': './data100',
     'num_classes': 100, 'proxy': 'resource/faat/proxy/resnet18_clean_cifar100.pth',
     'output': './resource/save_metric_100_res', 'budget': 1.5, 'seed': 1},
    {'name': 'icit_tiny_l2_2.0_seed1', 'dataset': 'tiny', 'data_dir': './data_tiny',
     'num_classes': 200, 'proxy': 'resource/faat/proxy/resnet18_clean_tiny.pth',
     'output': './resource/save_metric_tiny_res', 'budget': 2.0, 'seed': 1},
    {'name': 'icit_tiny_l2_1.5_seed1', 'dataset': 'tiny', 'data_dir': './data_tiny',
     'num_classes': 200, 'proxy': 'resource/faat/proxy/resnet18_clean_tiny.pth',
     'output': './resource/save_metric_tiny_res', 'budget': 1.5, 'seed': 1},
    {'name': 'icit_cifar10_l2_2.0_seed2', 'dataset': 'cifar10', 'data_dir': './data',
     'num_classes': 10, 'proxy': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
     'output': './resource/save_metric_10_res', 'budget': 2.0, 'seed': 2},
    {'name': 'icit_cifar10_l2_2.0_seed3', 'dataset': 'cifar10', 'data_dir': './data',
     'num_classes': 10, 'proxy': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
     'output': './resource/save_metric_10_res', 'budget': 2.0, 'seed': 3},
]


def free_gpus(min_free=8000, util_max=30):
    out = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits']).decode().strip().split('\n')
    free = []
    for line in out:
        idx, used, total, util = [int(x.strip()) for x in line.split(',')]
        if idx in EXCLUDE:
            continue
        if total - used >= min_free and util <= util_max:
            free.append(idx)
    return sorted(free)


def main():
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {j['name']: 'queued' for j in JOBS}
        json.dump(state, open(STATE, 'w'), indent=2)
    running = set()
    # detect already-running train_icit by checking result dir presence / log freshness
    print('[icit-disp] %d jobs; exclude GPUs %s' % (len(JOBS), sorted(EXCLUDE)))
    while any(v == 'queued' for v in state.values()):
        for j in JOBS:
            if state[j['name']] != 'queued':
                continue
            gpu = next((g for g in free_gpus() if g not in running), None)
            if gpu is None:
                break
            log = 'logs/icit_multidata/%s.log' % j['name']
            cmd = ('CUDA_VISIBLE_DEVICES=%d %s -m faat.train_icit --dataset %s --data_dir %s '
                   '--num_classes %d --proxy_path %s --train_proxy_if_missing '
                   '--save_trigger resource/faat/icit/%s --budget %g --poison_rate %g '
                   '--y_target 0 --seed %d --epochs 300 --steps 3000 '
                   '--output_dir %s --selection res --res_sel square --result_dir results/%s --gpu %d '
                   '> %s 2>&1' % (gpu, PY, j['dataset'], j['data_dir'], j['num_classes'], j['proxy'],
                                  j['name'], j['budget'],
                                  0.0025 if j['dataset'] == 'tiny' else 0.01,
                                  j['seed'], j['output'], j['name'], gpu, log))
            subprocess.Popen(['bash', '-c', cmd])
            running.add(gpu)
            state[j['name']] = 'running_gpu%d' % gpu
            json.dump(state, open(STATE, 'w'), indent=2)
            print('[icit-disp] launched %s on GPU%d' % (j['name'], gpu), flush=True)
            time.sleep(20)  # let it claim memory before next poll
        time.sleep(60)
        # reclaim GPUs whose job finished (result dir has model_last.pth)
        for j in JOBS:
            tag = state[j['name']]
            if tag.startswith('running_gpu'):
                g = int(tag.split('gpu')[1])
                if os.path.exists('results/%s/model_last.pth' % j['name']):
                    state[j['name']] = 'done'
                    json.dump(state, open(STATE, 'w'), indent=2)
                    running.discard(g)
                    print('[icit-disp] DONE %s (GPU%d freed)' % (j['name'], g), flush=True)
    print('[icit-disp] ALL JOBS DISPATCHED+DONE')


if __name__ == '__main__':
    main()
