"""Generate the v6c CIFAR-100 FAAT queue at 0.5% poison (方案 A, direct vs paper Table 2).

The paper's CIFAR-100 Table 2 uses 0.2%/0.5% poison. FAAT's headline v6 wave uses 1%
(all target-class samples) -- not directly comparable. This wave runs FAAT at the paper's
exact 0.5% poison so we can claim "FAAT > paper's published 0.5% best (Badnets-C Res-x²
85.06)" apples-to-apples. (δ_global optimization is poison-rate-independent, but each run
re-optimizes its own trigger under resource/faat/v6c/ to keep waves independent.)

Same PROVEN CIFAR-10 v4 recipe: CE + 8000-step self-contained Narcissus + fix_global +
adaptive_l2_max=0.15. poison_rate=0.005 (50% of the 500 target-class samples = 250).

Grid: CIFAR-100 x L2 {1.5,2.0,2.5} x seed{1,2,3} = 9 runs, Res-x², 0.5% poison, 300 ep.

Prerequisites (all already exist from v6 prep -- reused):
  * data100/                      -- CIFAR-100
  * resource/save_metric_100_res  -- cal_metric cifar100 seeds 1/2/3 (reused; selection now picks 250 not 500)
  * resource/faat/proxy/resnet18_clean_cifar100.pth  -- clean proxy (reused)

Run:  python -m faat.gen_queue_v6c_cifar100_faat05   # -> logs/v6c_cifar100_faat05/queue.json
      python -m faat.scheduler --queue logs/v6c_cifar100_faat05/queue.json \
             --log_dir logs/v6c_cifar100_faat05 --state logs/v6c_cifar100_faat05/scheduler_state.json \
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
POISON_RATE = 0.005   # <-- 方案 A: paper's 0.5% (vs v6's 1%)


def main():
    queue = []
    for l2 in L2S:
        for sd in SEEDS:
            name = 'cifar100_l2_%s_seed%d_p05' % (str(l2), sd)
            queue.append({
                'name': name,
                'dataset': CIFAR100['dataset'], 'data_dir': CIFAR100['data_dir'],
                'num_classes': CIFAR100['num_classes'], 'size': 32,
                'proxy_path': CIFAR100['proxy_path'], 'output_dir': CIFAR100['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': POISON_RATE,
                # CIFAR-10 v4 proven recipe
                'global_mode': 'from_scratch', 'l2_budget': l2,
                'global_steps': 8000, 'global_lr': 0.02, 'global_loss': 'ce',
                'adaptive_l2_max': 0.15, 'adaptive_l2_ratio': 0.0,
                'seed': sd,
                'save_trigger': 'resource/faat/v6c/cifar100/l2_%s_seed%d' % (str(l2), sd),
                'result_dir': 'results/faatb_v6c_%s' % name,
                'epochs': 300,
            })
    os.makedirs('logs/v6c_cifar100_faat05', exist_ok=True)
    out = 'logs/v6c_cifar100_faat05/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v6c_cifar100_faat05] wrote %d runs -> %s' % (len(queue), out))
    print('  方案A: FAAT @ poison=%.1f%% (paper Table2 setting), L2 %s x seeds %s'
          % (POISON_RATE * 100, L2S, SEEDS))


if __name__ == '__main__':
    main()
