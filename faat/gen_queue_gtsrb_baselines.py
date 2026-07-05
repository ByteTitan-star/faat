"""Generate the GTSRB baseline queue for the matched-stealth comparison.

The paper reports NO GTSRB experiments, so requirement-1 ('beat baseline') cannot
be claimed by a direct number on GTSRB. Instead we run our own GTSRB baselines
through the SAME train_backdoor.py pipeline (identical model/optim/selection/data)
and compare FAAT vs baseline at MATCHED stealth: at equal SSIM/L2, FAAT ASR should
exceed baseline ASR. This is the honest, stronger framing.

Baselines (all work on GTSRB via the ImageFolder branch, no dataset-specific artifact):
  * badnets  : 3x3 corner patch (--type 0:0:0). Visible, high-ASR reference.
  * blend    : hello-kitty full-image blend alpha=0.1 (--blend_size 32). Visible ref.
  * quantize : MultiBpp-RGB 24:28:8. The paper's flagship clean-label attack -- the
               DIRECT predecessor FAAT must beat. Closest to FAAT in stealth profile.

Grid: 3 attacks x seed{1,2,3} = 9 runs, all Res-x² selection, 1% poison, 300 epochs.

Run:  python -m faat.gen_queue_gtsrb_baselines   # -> logs/v5_baselines/queue.json
      python -m faat.scheduler --queue logs/v5_baselines/queue.json \
             --log_dir logs/v5_baselines --state logs/v5_baselines/scheduler_state.json \
             --exclude 1 --poll 20
"""
import os
import json

GTSRB = {
    'dataset': 'gtsrb', 'data_dir': 'data/GTSRB32', 'num_classes': 43,
    'output_dir': './resource/save_metric_gtsrb',
}
SEEDS = [1, 2, 3]

# (backdoor_type, display tag, attack-specific spec)
ATTACKS = [
    {'backdoor_type': 'badnets', 'tag': 'badnets', 'type': '0:0:0'},
    {'backdoor_type': 'blend', 'tag': 'blend', 'blend_size': 32},
    {'backdoor_type': 'quantize', 'tag': 'quantize_rgb', 'num_levels': '24:28:8'},
]


def main():
    queue = []
    for atk in ATTACKS:
        for sd in SEEDS:
            name = 'gtsrb_%s_seed%d' % (atk['tag'], sd)
            spec = {
                'name': name, 'kind': 'baseline',
                'dataset': GTSRB['dataset'], 'data_dir': GTSRB['data_dir'],
                'num_classes': GTSRB['num_classes'], 'output_dir': GTSRB['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': 0.01, 'select_epoch': 10, 'seed': sd, 'epochs': 300,
                'backdoor_type': atk['backdoor_type'],
                'result_dir': 'results/%s' % name,
            }
            spec.update({k: v for k, v in atk.items() if k not in ('backdoor_type', 'tag')})
            queue.append(spec)
    os.makedirs('logs/v5_baselines', exist_ok=True)
    out = 'logs/v5_baselines/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_gtsrb_baselines] wrote %d GTSRB baseline runs -> %s'
          % (len(queue), out))
    print('  attacks: %s x seeds %s (Res-x², 1%% poison, 300 ep)'
          % ([a['tag'] for a in ATTACKS], SEEDS))


if __name__ == '__main__':
    main()
