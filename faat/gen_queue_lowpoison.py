"""B: ultra-low poison-rate robustness probe. Does Narcissus hold ASR at 0.1% / 0.05%?

If ASR collapses at low poison, 'a robust low-poison invisible trigger' is a novelty
opening (setting where Narcissus fails). CIFAR-10, v4 recipe, L2=1.5, seed1, 300ep.
"""
import os, json

C10 = {'dataset': 'cifar10', 'data_dir': './data', 'num_classes': 10,
       'proxy_path': 'resource/faat/proxy/resnet18_clean_cifar10.pth',
       'output_dir': './resource/save_metric_10_res'}
queue = []
for pr, tag in [(0.001, 'p01'), (0.0005, 'p005')]:
    name = 'cifar10_l2_1.5_seed1_%s' % tag
    queue.append({
        'name': name, 'dataset': C10['dataset'], 'data_dir': C10['data_dir'],
        'num_classes': C10['num_classes'], 'size': 32, 'proxy_path': C10['proxy_path'],
        'output_dir': C10['output_dir'], 'y_target': 0, 'selection': 'res', 'res_sel': 'square',
        'poison_rate': pr, 'global_mode': 'from_scratch', 'l2_budget': 1.5,
        'global_steps': 8000, 'global_lr': 0.02, 'global_loss': 'ce',
        'adaptive_l2_max': 0.15, 'adaptive_l2_ratio': 0.0, 'lambda_align': 1.0,
        'seed': 1, 'save_trigger': 'resource/faat/lowpoison/%s' % name,
        'result_dir': 'results/lowpoison_%s' % name, 'epochs': 300})
os.makedirs('logs/lowpoison', exist_ok=True)
json.dump(queue, open('logs/lowpoison/queue.json', 'w'), indent=2)
print('[lowpoison] wrote %d runs (0.1%%, 0.05%% poison)' % len(queue))
