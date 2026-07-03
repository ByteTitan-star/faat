#!/bin/bash
# run_faat_v3_1.sh — v3.1: 修 v3 的致命缺陷(adaptive 无界|da|~5 淹没 δ_global 破坏C2)。
# δ_global 冻结=Narcissus缩放(--fix_global) + δ_adaptive 硬约束 |da|<=0.15(--adaptive_l2_max)，
# 使 adaptive 只是轻度塑形扰动、绝不喧宾夺主，从而 victim 必须学 δ_global(测试期触发器)。
# 6 scale 档并行(GPU0/1/2 各2)。预期:ASR 追随 Narcissus + adaptive 提供有限 SS-规避塑形。
# 用法: nohup bash run_faat_v3_1.sh >/dev/null 2>&1 &   tail -f results/_run_faat_v3_1.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_v3_1.log
mkdir -p results

PIDF=results/_run_faat_v3_1.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 v3.1 实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT v3.1 sweep 启动(δ_global冻结 + |da|<=0.15, 6档并行) ====" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

CONFIGS="1.0|0 0.5|0 0.3|1 0.2|1 0.15|2 0.1|2"
COMMON="--dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --steps 2000 --batch_size 48 \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --train"

for c in $CONFIGS; do
  SC="${c%%|*}"; GPU="${c##*|}"
  ST="./resource/faat/v3_1/scale_${SC}"; RD="results/faatb_v3_1_scale_${SC}"; mkdir -p "$RD"
  echo "[$(date '+%T')] 启动 v3.1_scale_${SC} on GPU${GPU} -> $RD" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_faat.py $COMMON \
    --init_global_scale "$SC" --save_trigger "$ST" --gpu "$GPU" --result_dir "$RD" > "${RD}.stdout" 2>&1 &
  sleep 15
done

echo "[$(date '+%T')] 6 配置已并行启动，等待全部完成..." | tee -a "$LOG"
wait
echo "[$(date '+%F %T')] ===================== FAAT V3.1 SWEEP DONE =====================" | tee -a "$LOG"
