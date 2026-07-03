"""FAAT Stage B top-level orchestrator.

  1. Offline-optimise the trigger (delta_global + pi_theta + UnetGenerator) against
     the frozen clean proxy + c_target  -> writes artifacts to --save_trigger.
  2. (optional, --train) hand off to train_backdoor.py --backdoor_type faatb, which
     eager-injects the artifacts and trains+evaluates the victim model.

Example (optimise then train the victim on GPU2):
  CUDA_VISIBLE_DEVICES=2 python -u train_faat.py \
      --dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
      --device cuda --train_proxy_if_missing --proxy_epochs 100 \
      --save_trigger ./resource/faat/save_trigger_10_0 --steps 2000 \
      --train --gpu 2 --result_dir results/faatb_res_square
"""
import os
import sys
import subprocess

from faat.optimize import run_optimization, build_argparser


def main():
    parser = build_argparser()
    parser.add_argument('--train', action='store_true',
                        help='after optimisation, launch train_backdoor.py --backdoor_type faatb')
    parser.add_argument('--result_dir', default='results/faatb_res_square')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--inject_global_scale', type=float, default=1.0,
                        help='scale applied to the LEARNED delta_global at injection '
                             '(<1 for a stealth sweep; 1 = use as optimised)')
    parser.add_argument('--gpu', default=None,
                        help='CUDA_VISIBLE_DEVICES for the victim-training handoff')
    args = parser.parse_args()

    save_trigger, meta = run_optimization(args)
    print('[train_faat] optimisation done -> %s' % save_trigger)
    print('[train_faat] final align=%.4f  align_global=%.4f'
          % (meta['final_align'], meta['final_align_global']))

    if args.train:
        common = (
            "--dataset %s --model %s --epochs %d --learning_rate 0.1 --seed %d "
            "--y_target %d --poison_rate %g --output_dir %s --select_epoch %d "
            "--selection %s --res_sel %s --backdoor_type faatb "
            "--faat_global_scale %g --faat_save_trigger %s --result_dir %s "
            "--num_classes %d --data_dir %s"
        ) % (args.dataset, args.model, args.epochs, args.seed, args.y_target,
             args.poison_rate, args.output_dir, args.select_epoch,
             args.selection, args.res_sel, args.inject_global_scale,
             args.save_trigger, args.result_dir,
             args.num_classes, args.data_dir)
        cmd = "%s -u train_backdoor.py %s" % (sys.executable, common)
        env = os.environ.copy()
        if args.gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print('[train_faat] launching victim training: %s' % cmd)
        subprocess.run(cmd, shell=True, env=env, check=False)


if __name__ == '__main__':
    main()
