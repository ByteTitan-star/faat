#!/bin/bash
# run_faat_pareto.sh — Stage B 隐蔽-ASR Pareto 前沿攻关(目标3:肉眼不可见+高ASR)。
# δ_global 硬 L2 预算 sweep，每档=离线优化(--global_l2_max 精确控隐蔽)+受害训练300ep。
# GPU 礼让：动态选空闲(≥8GB)卡，与他人共享不抢占；MAX_JOBS 并发上限。
# 优先级顺序：先跑交叉点档(L2=2.0/1.3)，再跑深隐蔽(0.9)，最后 3.0(最不关键)。
# 用法: nohup bash run_faat_pareto.sh >/dev/null 2>&1 &   看进度: tail -f results/_run_faat_pareto.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_pareto.log
MAX_JOBS=2          # 礼让：同时最多占2卡(与他人共享)
FREE_MEM_MB=8000
mkdir -p results

PIDF=results/_run_faat_pareto.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 Pareto 调度器实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT Pareto 隐蔽-ASR sweep 启动 ====" | tee -a "$LOG"

# 作业清单：L2预算（顺序=优先级）
JOBS="2.0 1.3 0.9 3.0"

running()   { ps -eo cmd | grep 'train_faat.py' | grep -v grep | grep -oE -- '--global_l2_max [0-9.]+' | sort -u | wc -l; }
is_running(){ ps -eo cmd | grep 'train_faat.py' | grep -v grep | grep -qF -- "--global_l2_max $1"; }
is_done()   { [ -f "results/faatb_l2_$1/output_1.log" ] && grep -qE '\] - 299[[:space:]]' "results/faatb_l2_$1/output_1.log"; }
free_gpu()  {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v thr="$FREE_MEM_MB" '{gsub(/ /,"",$2); if ($2+0 >= thr) print $1, $2}' | \
  sort -k2 -nr | head -n 1 | awk '{print $1}'
}

COMMON="--dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --steps 2000 --batch_size 48 \
  --init_global_scale 1.0 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_align_global 0.5 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --train"

for L2 in $JOBS; do
  if is_done "$L2";   then echo "[$(date '+%T')] 跳过(已完成): L2=$L2" | tee -a "$LOG"; continue; fi
  if is_running "$L2"; then echo "[$(date '+%T')] 跳过(运行中): L2=$L2" | tee -a "$LOG"; continue; fi
  while [ "$(running)" -ge "$MAX_JOBS" ]; do sleep 60; done
  while :; do GPU=$(free_gpu); [ -n "$GPU" ] && break; echo "[$(date '+%T')] 无空闲GPU，等 L2=$L2..." | tee -a "$LOG"; sleep 90; done
  ST="./resource/faat/pareto/l2_${L2}"; RD="results/faatb_l2_${L2}"; mkdir -p "$RD"
  echo "[$(date '+%T')] 启动 L2_max=$L2 on GPU$GPU -> $RD" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_faat.py $COMMON \
    --global_l2_max "$L2" --save_trigger "$ST" \
    --gpu "$GPU" --result_dir "$RD" > "${RD}.stdout" 2>&1 &
  sleep 20
done

while [ "$(running)" -gt 0 ]; do sleep 120; done
echo "[$(date '+%F %T')] ===================== FAAT PARETO SWEEP DONE =====================" | tee -a "$LOG"
