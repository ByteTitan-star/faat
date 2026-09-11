#!/bin/bash
# P0-2b KST-Learn: 1% poison, 60ep, 3 seeds.
# alpha sweep at eps=16/255: {0.0 (pure CE), 0.5, 2.0} x seeds {1,2,3}
# + rescue test: alpha=0.5 at eps=8/255 x seeds {1,2,3}  (pure KST eps=8 failed at ASR 0.03)
# 12 runs = 3 waves of 4 on GPU 0-3.
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PROXY=resource/faat/proxy/resnet18_clean_cifar10.pth

run() {  # alpha e seed gpu eps
  a=$1; e=$2; seed=$3; gpu=$4; eps=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY -u faat/kst_sdt_validate.py \
    --trigger kst --gpu 0 --epochs 60 --poison_rate 0.01 --eps $eps --seed $seed \
    --proxy_path $PROXY --kst_alpha $a --save_model \
    > results/kst_sdt/p02b_a${a}_e${e}_s${seed}.log 2>&1 &
  echo "  launched alpha=$a e=$e seed=$seed gpu=$gpu"
}

echo "=== WAVE 1 (a0.0 e16 s1/2/3 + a0.5 e16 s1) ==="
run 0.0 16 1 0 0.0627; run 0.0 16 2 1 0.0627; run 0.0 16 3 2 0.0627; run 0.5 16 1 3 0.0627
wait
echo "=== WAVE 2 (a0.5 e16 s2/3 + a2.0 e16 s1/2) ==="
run 0.5 16 2 0 0.0627; run 0.5 16 3 1 0.0627; run 2.0 16 1 2 0.0627; run 2.0 16 2 3 0.0627
wait
echo "=== WAVE 3 (a2.0 e16 s3 + a0.5 e8 s1/2/3 rescue) ==="
run 2.0 16 3 0 0.0627; run 0.5 8 1 1 0.0314; run 0.5 8 2 2 0.0314; run 0.5 8 3 3 0.0314
wait
echo "=== ALL 12 DONE ==="; date
