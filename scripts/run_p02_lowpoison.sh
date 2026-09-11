#!/bin/bash
# P0-2: 1% poison, 3 seeds, KST vs Narcissus at eps in {8,16}/255.
# 12 runs = 2 triggers x 2 eps x 3 seeds. 3 waves of 4 on GPU 0-3. 60 epochs.
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main

run() {
  trig=$1; e=$2; seed=$3; gpu=$4; eps=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY -u faat/kst_sdt_validate.py \
    --trigger $trig --gpu 0 --epochs 60 --poison_rate 0.01 --eps $eps --seed $seed --save_model \
    > results/kst_sdt/p02_${trig}_e${e}_s${seed}.log 2>&1 &
  echo "  launched $trig e=$e seed=$seed gpu=$gpu"
}

echo "=== WAVE 1 (kst e8 s1/2/3 + narc e8 s1) ==="
run kst 8 1 0 0.0314; run kst 8 2 1 0.0314; run kst 8 3 2 0.0314; run narcissus 8 1 3 0.0314
wait
echo "=== WAVE 2 (kst e16 s1/2/3 + narc e8 s2) ==="
run kst 16 1 0 0.0627; run kst 16 2 1 0.0627; run kst 16 3 2 0.0627; run narcissus 8 2 3 0.0314
wait
echo "=== WAVE 3 (narc e8 s3 + narc e16 s1/2/3) ==="
run narcissus 8 3 0 0.0314; run narcissus 16 1 1 0.0627; run narcissus 16 2 2 0.0627; run narcissus 16 3 3 0.0627
wait
echo "=== ALL 12 DONE ==="
date
