#!/bin/bash
# FEAST victim-training stage (4-GPU). Triggers already optimized & saved in resource/faat/feast/*,
# so this calls train_backdoor directly (skips re-optimization). 300 epochs each (~2.5h).
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/feast
COMMON="--dataset cifar10 --model resnet18 --backdoor_type feast --num_classes 10 --y_target 0 --poison_rate 0.01 --seed 1 --epochs 300 --output_dir ./resource/save_metric_10_res --select_epoch 10 --selection res --res_sel square"

CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m train_backdoor $COMMON \
  --feast_save_trigger resource/faat/feast/pix_b1.0_starve_s1 --feast_starve_eps 0.03137 \
  --result_dir results/feast_pix_b1.0_starve_s1 > logs/feast/pix_b1.0_starve_s1.log 2>&1 &
echo "GPU0 pixel b1.0 + starve PID=$!"
CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m train_backdoor $COMMON \
  --feast_save_trigger resource/faat/feast/pix_b1.0_nostarve_s1 --feast_starve_eps 0 \
  --result_dir results/feast_pix_b1.0_nostarve_s1 > logs/feast/pix_b1.0_nostarve_s1.log 2>&1 &
echo "GPU1 pixel b1.0 NO-starve (control) PID=$!"
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m train_backdoor $COMMON \
  --feast_save_trigger resource/faat/feast/pix_b0.5_starve_s1 --feast_starve_eps 0.03137 \
  --result_dir results/feast_pix_b0.5_starve_s1 > logs/feast/pix_b0.5_starve_s1.log 2>&1 &
echo "GPU2 pixel b0.5 + starve PID=$!"
CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m train_backdoor $COMMON \
  --feast_save_trigger resource/faat/feast/phase_starve_s1 --feast_starve_eps 0.03137 \
  --result_dir results/feast_phase_starve_s1 > logs/feast/phase_starve_s1.log 2>&1 &
echo "GPU3 phase + starve (user's design) PID=$!"
wait
echo "=== FEAST TRAINING DONE ==="
