"""Generate the Tiny-ImageNet FAAT queue at 0.25% poison (paper Table 7 setting).

Tiny-ImageNet: 200 classes, 100k train images (32x32, pre-resized into data_tiny/).
Paper Table 7 uses 0.25% poison (250 samples = 50% of the 500-sample target class).
Paper's best baseline ASR is only 43.93 (Blended-C Res-x^2) -- 200 classes at 0.25%
is hard for visible triggers. FAAT's invisible trigger + feature alignment should
clear this by a wide margin (as it did on CIFAR-10/100/GTSRB).

Same PROVEN v4 recipe as CIFAR-100 v6c: CE + 8000-step self-contained Narcissus +
fix_global + adaptive_l2_max=0.15. poison_rate=0.0025.

Grid: Tiny-ImageNet x L2 {1.5,2.0,2.5} x seed{1,2,3} = 9 runs, Res-x^2, 0.25% poison, 300 ep.

Prerequisites:
  * data_tiny/                       -- Tiny-ImageNet 32x32 (ImageFolder train/ + val/)
  * resource/save_metric_tiny_res    -- cal_metric tiny seeds 1/2/3 (stats_forget + epoch_10 pkl)
  * resource/faat/proxy/resnet18_clean_tiny.pth  -- clean proxy (train_clean_proxy, 100 ep)

Run:  python -m faat.gen_queue_v7_tiny_faat   # -> logs/v7_tiny_faat/queue.json
      python -m faat.scheduler --queue logs/v7_tiny_faat/queue.json \
             --log_dir logs/v7_tiny_faat --state logs/v7_tiny_faat/scheduler_state.json \
             --exclude 1 --poll 20
"""
import os
import json

TINY = {
    'dataset': 'tiny', 'data_dir': './data_tiny', 'num_classes': 200,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_tiny.pth',
    'output_dir': './resource/save_metric_tiny_res',
}
SEEDS = [1, 2, 3]
L2S = [1.5, 2.0, 2.5]
POISON_RATE = 0.0025   # <-- paper Table 7 setting: 0.25% (250 of 100k)


def main():
    queue = []
    for l2 in L2S:
        for sd in SEEDS:
            name = 'tiny_l2_%s_seed%d_p025' % (str(l2), sd)
            queue.append({
                'name': name,
                'dataset': TINY['dataset'], 'data_dir': TINY['data_dir'],
                'num_classes': TINY['num_classes'], 'size': 32,
                'proxy_path': TINY['proxy_path'], 'output_dir': TINY['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': POISON_RATE,
                # proven CIFAR-10/100 v4 recipe
                'global_mode': 'from_scratch', 'l2_budget': l2,
                'global_steps': 8000, 'global_lr': 0.02, 'global_loss': 'ce',
                'adaptive_l2_max': 0.15, 'adaptive_l2_ratio': 0.0,
                'seed': sd,
                'save_trigger': 'resource/faat/v7/tiny/l2_%s_seed%d' % (str(l2), sd),
                'result_dir': 'results/faatb_v7_%s' % name,
                'epochs': 300,
            })
    os.makedirs('logs/v7_tiny_faat', exist_ok=True)
    out = 'logs/v7_tiny_faat/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v7_tiny_faat] wrote %d runs -> %s' % (len(queue), out))
    print('  FAAT @ poison=%.2f%% (paper Table7 setting), L2 %s x seeds %s'
          % (POISON_RATE * 100, L2S, SEEDS))


if __name__ == '__main__':
    main()
