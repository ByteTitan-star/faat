#!/bin/bash
# run_faat_stageA.sh — FAAT Stage A（规则版）ASR 验证。
# GPU 礼让：先等 run_table1.sh 跑完，再在空闲(≥8GB)卡上跑；跳过已完成/运行中；PID 单实例。
# 用法: nohup bash run_faat_stageA.sh >/dev/null 2>&1 &    看进度: tail -f results/_run_faat_stageA.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_stageA.log
COMMON="--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10 --selection res --res_sel square --backdoor_type faat"
MAX_JOBS=4
FREE_MEM_MB=8000   # 只占空闲≥8GB 的卡，避免抢占

mkdir -p results
PIDF=results/_run_faat_stageA.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 FAAT 调度器实例(PID $(cat "$PIDF"))在运行，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT Stage A 调度器启动（等 run_table1 结束后开跑）====" | tee -a "$LOG"

# 1) 等 run_table1.sh 结束（PID 文件消失或进程不在）
while [ -f results/_run_table1.pid ] && kill -0 "$(cat results/_run_table1.pid 2>/dev/null)" 2>/dev/null; do
  echo "[$(date '+%T')] 等待 run_table1.sh (PID $(cat results/_run_table1.pid)) 结束..." | tee -a "$LOG"
  sleep 300
done
echo "[$(date '+%F %T')] run_table1.sh 已结束，开始 FAAT Stage A" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

# 2) 作业列表：dir|extra_args  （顺序=优先级）
read -r -d '' JOBS <<'EOF'
faat_res_square_gs100|--faat_global_scale 1.0
faat_res_square_gs100_noadp|--faat_global_scale 1.0 --faat_eps 0
faat_res_square_gs050|--faat_global_scale 0.5
faat_res_square_gs010|--faat_global_scale 0.1
EOF

running()   { ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -oE -- '--result_dir results/[^ ]+' | sort -u | wc -l; }
is_running(){ ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -qF -- "--result_dir results/$1"; }
is_done()   { [ -f "results/$1/output_1.log" ] && grep -qE '\] - 299[[:space:]]' "results/$1/output_1.log"; }
free_gpu()  {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v thr="$FREE_MEM_MB" '{gsub(/ /,"",$2); if ($2+0 >= thr) print $1, $2}' | \
  sort -k2 -nr | head -n 1 | awk '{print $1}'
}

echo "$JOBS" | grep -vE '^[[:space:]]*#' | while IFS='|' read -r dir extra; do
  [ -z "$dir" ] && continue
  if is_done "$dir";   then echo "[$(date '+%T')] 跳过(已完成): $dir" | tee -a "$LOG"; continue; fi
  if is_running "$dir"; then echo "[$(date '+%T')] 跳过(运行中): $dir" | tee -a "$LOG"; continue; fi
  while [ "$(running)" -ge "$MAX_JOBS" ]; do sleep 60; done
  while :; do
    GPU=$(free_gpu)
    [ -n "$GPU" ] && break
    echo "[$(date '+%T')] 无空闲GPU(≥${FREE_MEM_MB}MiB)，等待 $dir..." | tee -a "$LOG"
    sleep 90
  done
  mkdir -p "results/$dir"
  echo "[$(date '+%T')] 启动 $dir on GPU$GPU  [faat $extra]" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_backdoor.py $COMMON $extra --result_dir "results/$dir" > "results/$dir.stdout" 2>&1 &
  sleep 20   # 错峰，便于 free_gpu 选到不同卡
done

# 3) 等全部 FAAT 训练结束
while [ "$(running)" -gt 0 ]; do sleep 120; done
echo "[$(date '+%F %T')] ===================== FAAT STAGE A DONE =====================" | tee -a "$LOG"
# 不跑 parse_detail --write-md（避免污染 result_all.md 的 Table 1 表）；faat 结果由 Claude 单独解析进 docs/FAAT-EXPERIMENTS.md
