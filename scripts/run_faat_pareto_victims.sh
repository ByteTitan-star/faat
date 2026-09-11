#!/bin/bash
# run_faat_pareto_victims.sh — 用【正确的 Pareto artifact】重跑 victim 训练。
# 修 bug 后:--faat_save_trigger 指向 resource/faat/pareto/l2_X(各自优化的小L2 δ_global),
#   而非之前硬编码的 base(save_trigger_10_0)。artifact 已存在,只跑 victim(300ep),不重新优化。
# GPU 礼让:动态选空闲(≥8GB)卡，MAX_JOBS=2。
# 用法: nohup bash run_faat_pareto_victims.sh >/dev/null 2>&1 &   tail -f results/_run_faat_pareto_victims.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_pareto_victims.log
MAX_JOBS=2
FREE_MEM_MB=8000
mkdir -p results

PIDF=results/_run_faat_pareto_victims.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT Pareto victim 重跑(正确 artifact)启动 ====" | tee -a "$LOG"

JOBS="2.0 1.3 0.9 3.0"
running()   { ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -oE -- 'faat_save_trigger [^ ]+' | sort -u | wc -l; }
is_running(){ ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -qF -- "l2_$1"; }
is_done()   { [ -f "results/faatb_l2_$1/output_1.log" ] && grep -qE '\] - 299[[:space:]]' "results/faatb_l2_$1/output_1.log"; }
free_gpu()  {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v thr="$FREE_MEM_MB" '{gsub(/ /,"",$2); if ($2+0 >= thr) print $1, $2}' | \
  sort -k2 -nr | head -n 1 | awk '{print $1}'
}

COMMON="--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 \
  --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10 \
  --selection res --res_sel square --backdoor_type faatb --faat_global_scale 1.0"

for L2 in $JOBS; do
  if is_done "$L2";   then echo "[$(date '+%T')] 跳过(已完成): L2=$L2" | tee -a "$LOG"; continue; fi
  if is_running "$L2"; then echo "[$(date '+%T')] 跳过(运行中): L2=$L2" | tee -a "$LOG"; continue; fi
  while [ "$(running)" -ge "$MAX_JOBS" ]; do sleep 60; done
  while :; do GPU=$(free_gpu); [ -n "$GPU" ] && break; echo "[$(date '+%T')] 无空闲GPU，等 L2=$L2..." | tee -a "$LOG"; sleep 90; done
  ST="./resource/faat/pareto/l2_${L2}"; RD="results/faatb_l2_${L2}"; mkdir -p "$RD"
  echo "[$(date '+%T')] 启动 L2=$L2 on GPU$GPU  artifact=$ST" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_backdoor.py $COMMON \
    --faat_save_trigger "$ST" --result_dir "$RD" > "${RD}.stdout" 2>&1 &
  sleep 20
done

while [ "$(running)" -gt 0 ]; do sleep 120; done
echo "[$(date '+%F %T')] ===================== FAAT PARETO VICTIMS DONE =====================" | tee -a "$LOG"
