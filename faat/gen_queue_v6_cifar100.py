"""Generate the v6 CIFAR-100 FAAT queue -- third dataset (paper has CIFAR-100 Table 2).

Uses the PROVEN CIFAR-10 v4 recipe (CE + 8000-step self-contained Narcissus +
fix_global + adaptive_l2_max=0.15), which on CIFAR-10 gave ASR 93.6 > paper 84.88
with full defense evasion. CIFAR-100 (100 classes) needs a larger L2 than CIFAR-10
(10 classes) for equivalent fooling, so the sweep is shifted up: {1.5, 2.0, 2.5}.

Prerequisites (run before the scheduler):
  * data100/                      -- CIFAR-100 (hf-mirror download)
  * resource/save_metric_100_res  -- cal_metric.py --dataset cifar100 --epochs 11 (per seed)
  * resource/faat/proxy/resnet18_clean_cifar100.pth  -- train_clean_proxy (100 ep)

Grid: CIFAR-100 x L2 {1.5,2.0,2.5} x seed{1,2,3} = 9 runs, Res-x², 1% poison, 300 ep.

Run:  python -m faat.gen_queue_v6_cifar100   # -> logs/v6_cifar100/queue.json
      python -m faat.scheduler --queue logs/v6_cifar100/queue.json \
             --log_dir logs/v6_cifar100 --state logs/v6_cifar100/scheduler_state.json \
             --exclude 1 --poll 20
"""
import os
import json

CIFAR100 = {
    'dataset': 'cifar100', 'data_dir': './data100', 'num_classes': 100,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_cifar100.pth',
    'output_dir': './resource/save_metric_100_res',
}
SEEDS = [1, 2, 3]
L2S = [1.5, 2.0, 2.5]


def main():
    queue = []
    for l2 in L2S:
        for sd in SEEDS:
            name = 'cifar100_l2_%s_seed%d' % (str(l2), sd)
            queue.append({
                'name': name,
                'dataset': CIFAR100['dataset'], 'data_dir': CIFAR100['data_dir'],
                'num_classes': CIFAR100['num_classes'], 'size': 32,
                'proxy_path': CIFAR100['proxy_path'], 'output_dir': CIFAR100['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': 0.01,
                # CIFAR-10 v4 proven recipe: CE + 8000-step self-contained Narcissus
                'global_mode': 'from_scratch', 'l2_budget': l2,
                'global_steps': 8000, 'global_lr': 0.02, 'global_loss': 'ce',
                # fix_global + bounded adaptive (absolute 0.15, same as CIFAR-10 v4)
                'adaptive_l2_max': 0.15, 'adaptive_l2_ratio': 0.0,
                'seed': sd,
                'save_trigger': 'resource/faat/v6/cifar100/l2_%s_seed%d' % (str(l2), sd),
                'result_dir': 'results/faatb_v6_%s' % name,
                'epochs': 300,
            })
    os.makedirs('logs/v6_cifar100', exist_ok=True)
    out = 'logs/v6_cifar100/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v6_cifar100] wrote %d runs -> %s' % (len(queue), out))
    print('  CIFAR-10 v4 recipe: CE+8000 + fix_global + adaptive0.15; L2:', L2S, 'x seeds:', SEEDS)


if __name__ == '__main__':
    main()
