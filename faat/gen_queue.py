"""Generate the full v4 sweep queue (self-contained FAAT, multi-dataset + stealth
sweep + multi-seed) for the GPU-aware scheduler. Writes logs/v4/queue_full.json.

CIFAR-10 grid: L2 in {0.9, 1.2, 1.5} x seed in {1,2,3}
GTSRB   grid: L2 in {2.5, 3.0, 3.5} x seed in {1,2,3}

Edit GRIDS below to change. The scheduler processes the list in order and skips
no-one -- so remove/keep rows as desired before launching.

Run:  python -m faat.gen_queue          # -> logs/v4/queue_full.json
      python -m faat.scheduler --queue logs/v4/queue_full.json --exclude 1 --poll 20
"""
import os
import json

CIFAR = {
    'dataset': 'cifar10', 'data_dir': './data', 'num_classes': 10,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
    'output_dir': './resource/save_metric_10_res',
    'global_steps': 8000, 'l2s': [0.9, 1.2, 1.5],
}
GTSRB = {
    'dataset': 'gtsrb', 'data_dir': 'data/GTSRB32', 'num_classes': 43,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_gtsrb.pth',
    'output_dir': './resource/save_metric_gtsrb',
    'global_steps': 10000, 'l2s': [2.5, 3.0, 3.5],
}
SEEDS = [1, 2, 3]


def main():
    queue = []
    for tag, g in [('cifar10', CIFAR), ('gtsrb', GTSRB)]:
        for l2 in g['l2s']:
            for sd in SEEDS:
                name = '%s_l2_%s_seed%d' % (tag, str(l2), sd)
                queue.append({
                    'name': name, 'dataset': g['dataset'], 'data_dir': g['data_dir'],
                    'num_classes': g['num_classes'], 'size': 32,
                    'proxy_path': g['proxy_path'], 'output_dir': g['output_dir'],
                    'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                    'poison_rate': 0.01, 'l2_budget': l2,
                    'global_steps': g['global_steps'], 'seed': sd,
                    'save_trigger': 'resource/faat/v4/%s/l2_%s_seed%d' % (tag, str(l2), sd),
                    'result_dir': 'results/faatb_v4_%s' % name,
                    'epochs': 300,
                })
    os.makedirs('logs/v4', exist_ok=True)
    out = 'logs/v4/queue_full.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue] wrote %d experiments -> %s' % (len(queue), out))


if __name__ == '__main__':
    main()
