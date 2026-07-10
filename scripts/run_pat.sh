#!/bin/bash
# PAT (Perceptually-Allocated Trigger) 4-GPU sweep: 3 alphas + Narcissus baseline.
# Does perceptual (JND) budget allocation beat uniform-L2 (Narcissus) at matched stealth?
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pat

CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_pat --alpha 1.0 --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/pat_a1.0_s1 --save_trigger resource/faat/pat/a1.0 > logs/pat/a1.0.log 2>&1 &
echo "GPU0 PAT alpha=1.0 PID=$!"
CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_pat --alpha 0.5 --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/pat_a0.5_s1 --save_trigger resource/faat/pat/a0.5 > logs/pat/a0.5.log 2>&1 &
echo "GPU1 PAT alpha=0.5 PID=$!"
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m faat.train_pat --alpha 1.5 --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/pat_a1.5_s1 --save_trigger resource/faat/pat/a1.5 > logs/pat/a1.5.log 2>&1 &
echo "GPU2 PAT alpha=1.5 PID=$!"
# Narcissus baseline (authors' uniform universal noise) on GPU3, for PAT comparison
CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m train_backdoor --dataset cifar10 --model resnet18 \
  --backdoor_type narcissus --num_classes 10 --y_target 0 --poison_rate 0.01 --seed 1 --epochs 300 \
  --result_dir results/narcissus_baseline_s1 --output_dir ./resource/save_metric_10_res \
  --select_epoch 10 --selection res --res_sel square > logs/pat/narcissus_baseline.log 2>&1 &
echo "GPU3 Narcissus baseline PID=$!"
wait
echo "=== PAT SWEEP DONE ==="
