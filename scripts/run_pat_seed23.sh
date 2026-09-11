#!/bin/bash
# Multi-seed (2,3) confirmation: PAT alpha=0.5 (stealthy winner) vs Narcissus, to confirm
# PAT dominates Narcissus on stealth-ASR Pareto across seeds + 3 goals robust.
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pat

CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_pat --alpha 0.5 --y_target 0 --seed 2 --steps 3000 \
  --result_dir results/pat_a0.5_s2 --save_trigger resource/faat/pat/a0.5_s2 > logs/pat/a0.5_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_pat --alpha 0.5 --y_target 0 --seed 3 --steps 3000 \
  --result_dir results/pat_a0.5_s3 --save_trigger resource/faat/pat/a0.5_s3 > logs/pat/a0.5_s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m train_backdoor --dataset cifar10 --model resnet18 \
  --backdoor_type narcissus --num_classes 10 --y_target 0 --poison_rate 0.01 --seed 2 --epochs 300 \
  --result_dir results/narcissus_baseline_s2 --output_dir ./resource/save_metric_10_res \
  --select_epoch 10 --selection res --res_sel square > logs/pat/narcissus_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m train_backdoor --dataset cifar10 --model resnet18 \
  --backdoor_type narcissus --num_classes 10 --y_target 0 --poison_rate 0.01 --seed 3 --epochs 300 \
  --result_dir results/narcissus_baseline_s3 --output_dir ./resource/save_metric_10_res \
  --select_epoch 10 --selection res --res_sel square > logs/pat/narcissus_s3.log 2>&1 &
echo "launched PAT a0.5 s2/s3 + Narcissus s2/s3"
wait
echo "=== DONE ==="