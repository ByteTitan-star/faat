"""Generate the L_align ablation queue: FAAT with lambda_align=0 (NO feature alignment)
vs the existing with-L_align runs. Tests whether L_align CAUSALLY drives AC/SS evasion.

Recipe is IDENTICAL to v4/v6/v7 except lambda_align=0.0 (was 1.0). With lambda_align=0:
  * delta_global still optimised via Narcissus CE (global_loss=ce, lambda_asr_global path)
  * the per-sample adaptive delta is shaped but WITHOUT the ||f(x')-c_target||_2 term
=> this isolates whether the feature-alignment C-mechanism is what breaks AC/SS,
   or whether Narcissus-structure alone already evades.

Compare AC/SS to existing:
  CIFAR-10 with-L_align (v4, l2_1.5):  AC~0.25, SS~0.43
  Tiny    with-L_align (v7, l2_2.0):   (in current chain, eval after)

Runs (lambda_align=0):
  cifar10 l2_1.5 x seed{1,2,3}   -- matches v4 l2_1.5 exactly (clean paired comparison)
  tiny    l2_2.0 x seed{1}        -- matches v7 l2_2.0 seed1

Run on GPU1 only (0/2/3 busy with Tiny chain):
  python -m faat.scheduler --queue logs/ablation_noalign/queue.json \
      --log_dir logs/ablation_noalign --state logs/ablation_noalign/scheduler_state.json \
      --exclude 0,2,3 --poll 20
"""
import os
import json

CIFAR10 = {'dataset': 'cifar10', 'data_dir': './data', 'num_classes': 10,
           'proxy_path': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
           'output_dir': './resource/save_metric_10_res'}
TINY = {'dataset': 'tiny', 'data_dir': './data_tiny', 'num_classes': 200,
        'proxy_path': 'resource/faat/proxy/resnet18_clean_tiny.pth',
        'output_dir': './resource/save_metric_tiny_res'}


def spec(ds_cfg, l2, sd, size=32, poison=0.01):
    name = '%s_l2_%s_seed%d_noalign' % (ds_cfg['dataset'], str(l2), sd)
    return {
        'name': name,
        'dataset': ds_cfg['dataset'], 'data_dir': ds_cfg['data_dir'],
        'num_classes': ds_cfg['num_classes'], 'size': size,
        'proxy_path': ds_cfg['proxy_path'], 'output_dir': ds_cfg['output_dir'],
        'y_target': 0, 'selection': 'res', 'res_sel': 'square',
        'poison_rate': poison,
        'global_mode': 'from_scratch', 'l2_budget': l2,
        'global_steps': 8000, 'global_lr': 0.02, 'global_loss': 'ce',
        'adaptive_l2_max': 0.15, 'adaptive_l2_ratio': 0.0,
        'lambda_align': 0.0,            # <-- THE ABLATION: feature alignment OFF
        'seed': sd,
        'save_trigger': 'resource/faat/ablation_noalign/%s/l2_%s_seed%d'
                        % (ds_cfg['dataset'], str(l2), sd),
        'result_dir': 'results/ablation_%s' % name,
        'epochs': 300,
    }


def main():
    queue = [spec(CIFAR10, 1.5, 1), spec(CIFAR10, 1.5, 2), spec(CIFAR10, 1.5, 3),
             spec(TINY, 2.0, 1, poison=0.0025)]
    os.makedirs('logs/ablation_noalign', exist_ok=True)
    out = 'logs/ablation_noalign/queue.json'
    json.dump(queue, open(out, 'w'), indent=2)
    print('[gen_queue_ablation_noalign] wrote %d runs -> %s' % (len(queue), out))
    print('  lambda_align=0.0 (feature alignment OFF) -- isolates C-mechanism vs Narcissus-structure')


if __name__ == '__main__':
    main()
