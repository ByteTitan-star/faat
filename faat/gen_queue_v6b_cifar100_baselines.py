"""Generate the CIFAR-100 BASELINE queue at 1% poison (方案 B, matched comparison).

FAAT runs at 1% poison (queue v6). To make req① ("beat baseline") FAIR, we run the
paper's visible-trigger baselines through the SAME train_backdoor.py pipeline at the
SAME 1% poison / ResNet18 / Res-x² / y_target=0 / 300ep -- apples-to-apples vs FAAT.
(Mirror of gen_queue_gtsrb_baselines.py, the GTSRB matched-baseline approach.)

Note: at 1% poison on CIFAR-100, 500 samples = 100% of the target class are selected,
so the selection strategy is moot here -- what matters is the trigger type. The paper's
Table 2 uses 0.2%/0.5% (selection matters there); this 1% wave is the matched-to-FAAT
comparison, complementing 方案 A (FAAT @ 0.5%, direct vs paper Table 2).

Baselines:
  * badnets   : 3x3 checkerboard (--type 0:0:0). Paper Table 2 Badnets-C.
  * blend     : hello-kitty full-image blend (--blend_size 32). Paper Table 2 Blended-C.
  * quantize  : MultiBpp-RGB 24:28:8. Paper's flagship clean-label attack.

Grid: 3 attacks x seed{1,2,3} = 9 runs, Res-x² selection, 1% poison, 300 ep.

Run:  python -m faat.gen_queue_v6b_cifar100_baselines   # -> logs/v6b_cifar100_baselines/queue.json
      python -m faat.scheduler --queue logs/v6b_cifar100_baselines/queue.json \
             --log_dir logs/v6b_cifar100_baselines --state logs/v6b_cifar100_baselines/scheduler_state.json \
             --exclude 1 --poll 20
"""
import os
import json

CIFAR100 = {
    'dataset': 'cifar100', 'data_dir': './data100', 'num_classes': 100,
    'output_dir': './resource/save_metric_100_res',
}
SEEDS = [1, 2, 3]

ATTACKS = [
    {'backdoor_type': 'badnets', 'tag': 'badnets', 'type': '0:0:0'},
    {'backdoor_type': 'blend', 'tag': 'blend', 'blend_size': 32},
    {'backdoor_type': 'quantize', 'tag': 'quantize_rgb', 'num_levels': '24:28:8'},
]


def main():
    queue = []
    for atk in ATTACKS:
        for sd in SEEDS:
            name = 'bl_c100_%s_seed%d' % (atk['tag'], sd)
            spec = {
                'name': name, 'kind': 'baseline',
                'dataset': CIFAR100['dataset'], 'data_dir': CIFAR100['data_dir'],
                'num_classes': CIFAR100['num_classes'], 'output_dir': CIFAR100['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': 0.01, 'select_epoch': 10, 'seed': sd, 'epochs': 300,
                'backdoor_type': atk['backdoor_type'],
                'result_dir': 'results/%s' % name,
            }
            spec.update({k: v for k, v in atk.items() if k not in ('backdoor_type', 'tag')})
            queue.append(spec)
    os.makedirs('logs/v6b_cifar100_baselines', exist_ok=True)
    out = 'logs/v6b_cifar100_baselines/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v6b_cifar100_baselines] wrote %d CIFAR-100 baseline runs -> %s'
          % (len(queue), out))
    print('  attacks: %s x seeds %s (Res-x², 1%% poison, 300 ep) -- matched to FAAT v6'
          % ([a['tag'] for a in ATTACKS], SEEDS))


if __name__ == '__main__':
    main()
