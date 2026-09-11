#!/usr/bin/env bash
# run_ood_abc.sh — 48h mechanism check: A/B/C arms x {cifar10, gtsrb} x seeds
#
# Design (CLAUDE.md): --fix_global freezes delta_global at the A/B/C engine output
# (v3.1-style bounded-adaptive downstream) so the ONLY difference between arms is
# the SOURCE of delta_global:
#   a   = positive-only target alignment   (--ood_pool target    --ood_calib none)
#   b   = OOD as extra optimisation DATA   (--ood_pool target+ood --ood_calib none)
#   c   = OOD as explicit calibration REF  (--ood_pool target    --ood_calib cos)
# Reference anchors (no rerun needed): frozen-repo faatb v4_l2_1.5 CIFAR-10 = 93.6,
# faatb_gtsrb_l2_3.5 = 82.7 (full-FAAT numbers; treat as context, not arm).
#
# Usage:   run_ood_abc.sh <cifar10|gtsrb> <gpu> <seed> <arm> [arm...]
# Example: nohup bash scripts/run_ood_abc.sh cifar10 2 1 a b c > results_ood/_abc_cifar10_gpu2.log 2>&1 &
#          nohup bash scripts/run_ood_abc.sh gtsrb   3 1 a b c > results_ood/_abc_gtsrb_gpu3.log 2>&1 &
#
# Go/No-Go (48h, CLAUDE.md §硬约束): hard-regime (gtsrb) 3-seed stable —
#   GO:  same L2 budget -> ASR clearly up (c vs a/b), or same ASR at lower budget; BA intact.
#   NO-GO: gains only on cifar10 / only one OOD source / c ~= b / needs larger perturbation.

set -u
DS="$1"; GPU="$2"; SEED="$3"; shift 3
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd "$(dirname "$0")/.."

if [ "$DS" = "cifar10" ]; then
  COMMON="--dataset cifar10 --num_classes 10 --data_dir ./data \
    --selection res --res_sel square --poison_rate 0.01 --select_epoch 10 \
    --output_dir ./resource/save_metric_10_res \
    --proxy_path ./resource/faat/proxy/resnet18_clean_cifar10.pth \
    --ood_dataset cifar100 \
    --global_l2_budget 1.5 --global_steps 8000 --global_batch_size 128 \
    --steps 2000 --batch_size 48 --adaptive_l2_max 0.15 --eps_max 0.05"
elif [ "$DS" = "gtsrb" ]; then
  COMMON="--dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 \
    --selection res --res_sel square --poison_rate 0.01 --select_epoch 10 \
    --output_dir ./resource/save_metric_gtsrb \
    --proxy_path ./resource/faat/proxy/resnet18_clean_gtsrb.pth \
    --ood_dataset cifar10 \
    --global_l2_budget 3.5 --global_steps 8000 --global_batch_size 128 \
    --steps 2000 --batch_size 48 --adaptive_l2_max 0.15 --eps_max 0.05"
else
  echo "unknown dataset $DS"; exit 1
fi

for ARM in "$@"; do
  case "$ARM" in
    a) POOL="target";     CALIB="none" ;;
    b) POOL="target+ood"; CALIB="none" ;;
    c) POOL="target";     CALIB="cos"  ;;
    *) echo "unknown arm $ARM (a|b|c)"; continue ;;
  esac
  TAG="oodabc_${DS}_${ARM}_seed${SEED}"
  ST="./resource_ood/triggers/${TAG}"
  RD="./results_ood/${TAG}"
  if grep -qs "epoch=299\|Epoch 299" "$RD"/output_1.log 2>/dev/null; then
    echo "[run_ood_abc] $TAG already done, skip"
    continue
  fi
  echo "[run_ood_abc] $(date '+%F %T') launching $TAG on GPU$GPU (pool=$POOL calib=$CALIB)"
  CUDA_VISIBLE_DEVICES="$GPU" $PY -u train_faat.py $COMMON \
    --seed "$SEED" --y_target 0 \
    --global_mode ood --ood_pool "$POOL" --ood_calib "$CALIB" --ood_weight 1.0 \
    --fix_global \
    --save_trigger "$ST" --result_dir "$RD" --train --gpu "$GPU"
  echo "[run_ood_abc] $(date '+%F %T') finished $TAG (exit $?)"
done
echo "[run_ood_abc] all requested arms done for $DS seed$SEED"
