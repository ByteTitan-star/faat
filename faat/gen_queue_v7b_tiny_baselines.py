"""Generate the Tiny-ImageNet BASELINE queue at 0.25% poison (matched comparison).

Paper Table 7: at 0.25% poison on 200 classes, visible-trigger baselines are weak
(Badnets-C Res-x^2 38.96, Blended-C Res-x^2 43.93). To make req① ("beat baseline")
FAIR and on the SAME pipeline as FAAT v7, we run the paper's visible-trigger
baselines through the SAME train_backdoor.py at the SAME 0.25% poison / ResNet18 /
Res-x^2 / y_target=0 / 300ep -- apples-to-apples vs FAAT. (Mirror of CIFAR-100 v6b.)

Note: paper uses a 9x9 Badnets checker on 64x64; we use the repo's 3x3 checker on
32x32 (same pipeline as all other datasets). This is a matched self-run comparison,
not a literal reproduction of paper Table 7's 64x64 numbers (those live in
docs/PAPER-BASELINES.md).

Baselines:
  * badnets   : 3x3 checkerboard (--type 0:0:0). Paper Table 7 Badnets-C.
  * blend     : hello-kitty full-image blend (--blend_size 32). Paper Table 7 Blended-C.
  * quantize  : MultiBpp-RGB 24:28:8. Paper's flagship clean-label attack.

Grid: 3 attacks x seed{1,2,3} = 9 runs, Res-x^2 selection, 0.25% poison, 300 ep.

Prerequisites:
  * data_tiny/                       -- Tiny-ImageNet 32x32
  * resource/save_metric_tiny_res    -- cal_metric tiny seeds 1/2/3

Run:  python -m faat.gen_queue_v7b_tiny_baselines   # -> logs/v7b_tiny_baselines/queue.json
      python -m faat.scheduler --queue logs/v7b_tiny_baselines/queue.json \
             --log_dir logs/v7b_tiny_baselines --state logs/v7b_tiny_baselines/scheduler_state.json \
             --exclude 1 --poll 20
"""
import os
import json

TINY = {
    'dataset': 'tiny', 'data_dir': './data_tiny', 'num_classes': 200,
    'output_dir': './resource/save_metric_tiny_res',
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
            name = 'bl_tiny_%s_seed%d' % (atk['tag'], sd)
            spec = {
                'name': name, 'kind': 'baseline',
                'dataset': TINY['dataset'], 'data_dir': TINY['data_dir'],
                'num_classes': TINY['num_classes'], 'output_dir': TINY['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': 0.0025, 'select_epoch': 10, 'seed': sd, 'epochs': 300,
                'backdoor_type': atk['backdoor_type'],
                'result_dir': 'results/%s' % name,
            }
            spec.update({k: v for k, v in atk.items() if k not in ('backdoor_type', 'tag')})
            queue.append(spec)
    os.makedirs('logs/v7b_tiny_baselines', exist_ok=True)
    out = 'logs/v7b_tiny_baselines/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v7b_tiny_baselines] wrote %d Tiny-ImageNet baseline runs -> %s'
          % (len(queue), out))
    print('  attacks: %s x seeds %s (Res-x^2, 0.25%% poison, 300 ep) -- matched to FAAT v7'
          % ([a['tag'] for a in ATTACKS], SEEDS))


if __name__ == '__main__':
    main()
