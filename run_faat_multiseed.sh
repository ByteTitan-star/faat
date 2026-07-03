#!/bin/bash
# run_faat_multiseed.sh — v3.1 多种子稳定性(scale 0.17/0.2 × seed 2/3 = 4 配置)。
# seed2/3 pkl 已由 cal_metric 生成。每配置 = train_faat 全流程(optimize adaptive for 该seed poison集 + victim)。
# δ_global=Narcissus 冻结(fix_global,跨seed不变),adaptive 按 seed 各自优化。GPU0/2 各2组。
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_multiseed.log
mkdir -p results
PIDF=results/_run_faat_multiseed.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then echo "已有实例，退出" | tee -a "$LOG"; exit 0; fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT
echo "[$(date '+%F %T')] ==== v3.1 多种子 sweep 启动(4配置) ====" | tee -a "$LOG"

V31="--dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --device cuda \
  --train_proxy_if_missing --steps 2000 --batch_size 48 \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 --train"

# GPU0: seed2 两档
for SC in 0.2 0.17; do
  CUDA_VISIBLE_DEVICES=0 nohup $PY -u train_faat.py $V31 --seed 2 --init_global_scale $SC \
    --save_trigger ./resource/faat/v3_1/scale_${SC}_seed2 --gpu 0 \
    --result_dir results/faatb_v3_1_scale_${SC}_seed2 > results/faatb_v3_1_scale_${SC}_seed2.stdout 2>&1 &
  echo "[$(date '+%T')] GPU0: scale$SC seed2" | tee -a "$LOG"; sleep 12
done
# GPU2: seed3 两档
for SC in 0.2 0.17; do
  CUDA_VISIBLE_DEVICES=2 nohup $PY -u train_faat.py $V31 --seed 3 --init_global_scale $SC \
    --save_trigger ./resource/faat/v3_1/scale_${SC}_seed3 --gpu 2 \
    --result_dir results/faatb_v3_1_scale_${SC}_seed3 > results/faatb_v3_1_scale_${SC}_seed3.stdout 2>&1 &
  echo "[$(date '+%T')] GPU2: scale$SC seed3" | tee -a "$LOG"; sleep 12
done
wait
echo "[$(date '+%F %T')] ===================== FAAT MULTISEED DONE =====================" | tee -a "$LOG"
