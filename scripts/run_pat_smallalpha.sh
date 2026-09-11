#!/usr/bin/env bash
# PAT small-alpha sweep: find highest-SSIM config with ASR>=0.86 (to Pareto-dominate BppAttack
# ASR 0.8388 / SSIM 0.935-0.949 / NC 6.77-caught). Existing: alpha=1.0 ASR 0.999/SSIM 0.763.
# alpha=0.3 trigger exists (SSIM 0.949) -> train directly. 0.2/0.15/0.1 need optimize+train.
set -u
cd "$(dirname "$0")/.."
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pat

# GPU0: alpha=0.3 (trigger exists -> direct train_backdoor with existing artifact)
CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m train_backdoor --dataset cifar10 --model resnet18 \
  --backdoor_type pat --num_classes 10 --y_target 0 --poison_rate 0.01 --seed 1 --epochs 300 \
  --pat_save_trigger resource/faat/pat/a0.3 --result_dir results/pat_a0.3_s1 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --selection res --res_sel square \
  > logs/pat/a0.3_train.stdout 2>&1 &
echo $! > logs/pat/a0.3_train.pid

# GPU1/2/3: alpha=0.2/0.15/0.1 (optimize+train via orchestrator)
i=1
for a in 0.2 0.15 0.1; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY -u -m faat.train_pat --alpha $a --steps 3000 --seed 1 \
    --save_trigger resource/faat/pat/a${a} --result_dir results/pat_a${a}_s1 \
    > logs/pat/a${a}_s1.stdout 2>&1 &
  echo $! > logs/pat/a${a}_s1.pid
  echo "launched alpha=$a on GPU$i (pid $(cat logs/pat/a${a}_s1.pid))"
  i=$((i+1))
done
echo "alpha=0.3 direct-train on GPU0 (pid $(cat logs/pat/a0.3_train.pid))"
