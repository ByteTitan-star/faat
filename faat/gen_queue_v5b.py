"""Generate the v5b GTSRB queue -- ASR recovery + L_align as the real AC/SS lever.

v5 verdict (from docs/v5_results.csv): CW-margin did NOT improve SSIM (SSIM is purely
L2-determined) and HURT ASR (lower proxyASR than CE); adaptive-l2-ratio=0.12 only
marginally helped AC/SS (0.66->0.64) at a large ASR cost (over-sized adaptive -> the
model over-relied on the train-only adaptive, a C2 co-trigger). So:

  * DROP CW -> back to CE (global_loss=ce, 10k steps = the proven v4 config).
  * Moderate adaptive (ratio=0.09 -> 0.225 at L2=2.5, between v4's 0.15 and v5's 0.30):
    more L_align room than v4 but not the v5 co-trigger extreme.
  * THE NEW LEVER: lambda_align 1.0 -> 3.0. v5 showed adaptive size is NOT the AC/SS
    bottleneck; the feature-embedding strength (L_align weight) is. Stronger pull
    embeds poison deeper into the target cluster -> lower AC/SS without growing the
    perturbation -> no ASR cost. This is the single-variable change vs v4.

Grid: GTSRB x L2 in {2.5, 3.0, 3.5} x seed{1,2,3} = 9 runs (the high-ASR region where
we need to push stealth; directly comparable to v4's {2.5,3.0,3.5}).

Run:  python -m faat.gen_queue_v5b          # -> logs/v5b/queue.json
      python -m faat.scheduler --queue logs/v5b/queue.json --log_dir logs/v5b \
             --state logs/v5b/scheduler_state.json --exclude 1 --poll 20
"""
import os
import json

GTSRB = {
    'dataset': 'gtsrb', 'data_dir': 'data/GTSRB32', 'num_classes': 43,
    'proxy_path': 'resource/faat/proxy/resnet18_clean_gtsrb.pth',
    'output_dir': './resource/save_metric_gtsrb',
}
SEEDS = [1, 2, 3]
L2S = [2.5, 3.0, 3.5]


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
                # CE (proven) + 10k steps; revert v5's CW
                'global_mode': 'from_scratch', 'l2_budget': l2,
                'global_steps': 10000, 'global_lr': 0.02,
                'global_loss': 'ce',
                # moderate adaptive scaling (ratio path; abs off)
                'adaptive_l2_max': 0.0, 'adaptive_l2_ratio': 0.09,
                # THE LEVER: 3x feature-alignment weight to embed poison -> drop AC/SS
                'lambda_align': 3.0,
                'seed': sd,
                'save_trigger': 'resource/faat/v5b/gtsrb/l2_%s_seed%d' % (str(l2), sd),
                'result_dir': 'results/faatb_v5b_%s' % name,
                'epochs': 300,
            })
    os.makedirs('logs/v5b', exist_ok=True)
    out = 'logs/v5b/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_v5b] wrote %d runs -> %s' % (len(queue), out))
    print('  CE+10k (revert CW) + adaptive_l2_ratio=0.09 + lambda_align=3.0 (the AC/SS lever)')
    print('  L2:', L2S, 'x seeds:', SEEDS)


if __name__ == '__main__':
    main()
