#!/bin/bash
# run_faat_batch2.sh — v3.1 后续验证批(按优先级)，GPU0/1/2 各2组(GPU3留同学)。
# P1消融(faat eps=0=纯Narcissus无adaptive) / P2微调sweep(0.17/0.18) / P3多y_target(1/2)。
# 显式 2/卡，直接启动+wait(不用 is_running，避免竞态)。~80min。
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_batch2.log
mkdir -p results

PIDF=results/_run_faat_batch2.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 batch2 实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT
echo "[$(date '+%F %T')] ==== FAAT batch2 启动(消融+微调+多y_target, 6配置并行) ====" | tee -a "$LOG"

UNI="--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 \
  --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10 --selection res --res_sel square"
V31="--dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --steps 2000 --batch_size 48 \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 --train"

# ---- GPU0: P1 消融 (纯Narcissus无adaptive, faat eps=0) ----
CUDA_VISIBLE_DEVICES=0 nohup $PY -u train_backdoor.py $UNI --y_target 0 \
  --backdoor_type faat --faat_global_scale 0.2 --faat_eps 0 \
  --result_dir results/ablation_scale0.2_noAdp > results/ablation_scale0.2_noAdp.stdout 2>&1 &
echo "[$(date '+%T')] GPU0: 消融 scale0.2 无adaptive" | tee -a "$LOG"; sleep 10
CUDA_VISIBLE_DEVICES=0 nohup $PY -u train_backdoor.py $UNI --y_target 0 \
  --backdoor_type faat --faat_global_scale 0.1 --faat_eps 0 \
  --result_dir results/ablation_scale0.1_noAdp > results/ablation_scale0.1_noAdp.stdout 2>&1 &
echo "[$(date '+%T')] GPU0: 消融 scale0.1 无adaptive" | tee -a "$LOG"; sleep 10

# ---- GPU1: P2 微调 sweep ----
CUDA_VISIBLE_DEVICES=1 nohup $PY -u train_faat.py $V31 --init_global_scale 0.17 \
  --save_trigger ./resource/faat/v3_1/scale_0.17 --gpu 1 --result_dir results/faatb_v3_1_scale_0.17 > results/faatb_v3_1_scale_0.17.stdout 2>&1 &
echo "[$(date '+%T')] GPU1: v3.1 scale0.17" | tee -a "$LOG"; sleep 10
CUDA_VISIBLE_DEVICES=1 nohup $PY -u train_faat.py $V31 --init_global_scale 0.18 \
  --save_trigger ./resource/faat/v3_1/scale_0.18 --gpu 1 --result_dir results/faatb_v3_1_scale_0.18 > results/faatb_v3_1_scale_0.18.stdout 2>&1 &
echo "[$(date '+%T')] GPU1: v3.1 scale0.18" | tee -a "$LOG"; sleep 10

# ---- GPU2: P3 多 y_target (scale 0.2) ----
CUDA_VISIBLE_DEVICES=2 nohup $PY -u train_faat.py $V31 --y_target 1 --init_global_scale 0.2 \
  --save_trigger ./resource/faat/v3_1/yt1_scale0.2 --gpu 2 --result_dir results/faatb_v3_1_yt1_scale0.2 > results/faatb_v3_1_yt1_scale0.2.stdout 2>&1 &
echo "[$(date '+%T')] GPU2: v3.1 y_target=1 scale0.2" | tee -a "$LOG"; sleep 10
CUDA_VISIBLE_DEVICES=2 nohup $PY -u train_faat.py $V31 --y_target 2 --init_global_scale 0.2 \
  --save_trigger ./resource/faat/v3_1/yt2_scale0.2 --gpu 2 --result_dir results/faatb_v3_1_yt2_scale0.2 > results/faatb_v3_1_yt2_scale0.2.stdout 2>&1 &
echo "[$(date '+%T')] GPU2: v3.1 y_target=2 scale0.2" | tee -a "$LOG"

wait
echo "[$(date '+%F %T')] ===================== FAAT BATCH2 DONE =====================" | tee -a "$LOG"
