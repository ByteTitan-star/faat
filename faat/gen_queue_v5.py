"""Generate the v5 GTSRB queue -- stealth-evasion fix sweep.

v5 targets the ONE weak requirement left after v4: GTSRB defense-evasion (AC/SS
0.66-0.80 detectable, SSIM 0.72-0.81). Two independent fixes (user-approved):

  * CW-margin global trigger + 20k steps (global_loss=cw, global_steps=20000):
    a more universal delta_global -> same ASR at LOWER L2 -> higher SSIM.
    (stealth metric is on delta_global only, per C2.)
  * adaptive budget scaled with ||delta_global|| (adaptive_l2_ratio=0.12,
    adaptive_l2_max=0): the L_align shaping budget was starved at a fixed 0.15
    (only ~5% of ||dg||~3 on GTSRB vs ~10% on CIFAR) -> AC/SS leaked. Scaling it
    lets L_align embed poison into the target cluster -> drive AC/SS down.

Grid: GTSRB x L2 in {1.5, 2.0, 2.5, 3.0} x seed in {1,2,3} = 12 runs.
  - Lower L2 than v4 ({2.5,3.0,3.5}) because CW should achieve equal ASR cheaper;
    3.0 kept as a direct v4 anchor for apples-to-apples stealth comparison.

Run:  python -m faat.gen_queue_v5          # -> logs/v5/queue.json
      python -m faat.scheduler --queue logs/v5/queue.json --log_dir logs/v5 \
             --state logs/v5/scheduler_state.json --exclude 1 --poll 20
"""
import os
import json

GTSRB = {
    'dataset': 'gtsrb', 'data_dir': 'data/GTSRB32', 'num_classes': 43,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_gtsrb.pth',
    'output_dir': './resource/save_metric_gtsrb',
}
SEEDS = [1, 2, 3]
L2S = [1.5, 2.0, 2.5, 3.0]


def main():
    queue = []
    for l2 in L2S:
        for sd in SEEDS:
            name = 'gtsrb_l2_%s_seed%d' % (str(l2), sd)
            queue.append({
                'name': name,
                'dataset': GTSRB['dataset'], 'data_dir': GTSRB['data_dir'],
                'num_classes': GTSRB['num_classes'], 'size': 32,
                'proxy_path': GTSRB['proxy_path'], 'output_dir': GTSRB['output_dir'],
                'y_target': 0, 'selection': 'res', 'res_sel': 'square',
                'poison_rate': 0.01,
                # v5 self-contained Narcissus: CW-margin + 20k steps (SSIM fix)
                'global_mode': 'from_scratch',
                'l2_budget': l2,
                'global_steps': 20000, 'global_lr': 0.02,
                'global_loss': 'cw', 'global_margin': 10.0,
                # v5 adaptive budget = 12% of ||delta_global|| (AC/SS fix);
                # absolute budget OFF so the ratio path takes over.
                'adaptive_l2_max': 0.0, 'adaptive_l2_ratio': 0.12,
                'seed': sd,
                'save_trigger': 'resource/faat/v5/gtsrb/l2_%s_seed%d' % (str(l2), sd),
                'result_dir': 'results/faatb_v5_%s' % name,
                'epochs': 300,
            })
    os.makedirs('logs/v5', exist_ok=True)
    out = 'logs/v5/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v5] wrote %d GTSRB v5 experiments -> %s' % (len(queue), out))
    print('  fixes: global_loss=cw global_steps=20000 (SSIM) + adaptive_l2_ratio=0.12 (AC/SS)')
    print('  L2 sweep:', L2S, 'x seeds:', SEEDS)


if __name__ == '__main__':
    main()
