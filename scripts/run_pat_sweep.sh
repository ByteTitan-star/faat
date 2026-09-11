#!/usr/bin/env bash
# PAT alpha-sweep: map ASR-SSIM Pareto, seek a config that DOMINATES BppAttack
# (BppAttack Res-x2: ASR 0.8388, our-measured SSIM 0.97, NC 6.77-caught).
# alpha=1.0 already done (ASR 0.98 / SSIM 0.93). Sweep smaller alpha (higher SSIM) + one larger.
# Target: any config with ASR>=0.86 AND SSIM>=0.97 -> clean Pareto win over BppAttack.
set -u
cd "$(dirname "$0")/.."
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pat

i=0
for a in 0.3 0.5 0.75 1.5; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY -u -m faat.train_pat --alpha $a --steps 3000 --seed 1 \
    --save_trigger resource/faat/pat/a${a} --result_dir results/pat_a${a}_s1 \
    > logs/pat/a${a}_s1.stdout 2>&1 &
  echo $! > logs/pat/a${a}_s1.pid
  echo "launched alpha=$a on GPU$i (pid $(cat logs/pat/a${a}_s1.pid))"
  i=$((i+1))
done
