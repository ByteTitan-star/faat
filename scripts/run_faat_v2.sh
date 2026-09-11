#!/bin/bash
# run_faat_v2.sh — v2 改进验证：δ_global 走 ASR-proxy 目标(--global_obj asr，Narcissus式)，
# L_align 只塑 δ_adaptive。6 配置并行(GPU0/1/2 各2个)，覆盖 base + 5 个隐蔽档。
# 目标：验证 v2 在等 L2 下 ASR 是否追平/超过 Narcissus、且仍保 SS-规避 → 能否同时达 T1+T3。
# 用法: nohup bash run_faat_v2.sh >/dev/null 2>&1 &   tail -f results/_run_faat_v2.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_v2.log
mkdir -p results

PIDF=results/_run_faat_v2.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 v2 实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT v2 sweep 启动(δ_global=ASR-proxy目标, 6配置并行) ====" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

# 格式: 名字|L2预算(0=无约束)|GPU
CONFIGS="base|0|0 l3.0|3.0|0 l2.0|2.0|1 l1.5|1.5|1 l1.3|1.3|2 l0.9|0.9|2"

COMMON="--dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --steps 2000 --batch_size 48 \
  --global_obj asr --lambda_asr_global 1.0 --init_global_scale 1.0 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --train"

for c in $CONFIGS; do
  NAME="${c%%|*}"; rest="${c#*|}"; L2="${rest%%|*}"; GPU="${rest##*|}"
  ST="./resource/faat/v2/${NAME}"; RD="results/faatb_v2_${NAME}"; mkdir -p "$RD"
  echo "[$(date '+%T')] 启动 v2_${NAME} (L2_max=$L2) on GPU${GPU} -> $RD" | tee -a "$LOG"
  if [ "$L2" = "0" ]; then L2ARG=""; else L2ARG="--global_l2_max $L2"; fi
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_faat.py $COMMON $L2ARG \
    --save_trigger "$ST" --gpu "$GPU" --result_dir "$RD" > "${RD}.stdout" 2>&1 &
  sleep 15
done

echo "[$(date '+%T')] 6 配置已并行启动，等待全部完成..." | tee -a "$LOG"
wait
echo "[$(date '+%F %T')] ===================== FAAT V2 SWEEP DONE =====================" | tee -a "$LOG"
