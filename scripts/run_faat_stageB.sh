#!/bin/bash
# run_faat_stageB.sh — FAAT Stage B（真 FAAT：特征对齐 + 摊还策略）GPU 执行。
# GPU 礼让：先等 run_faat_stageA.sh 与 run_table1.sh 都结束，再在空闲(≥8GB)卡上跑。
# 三步串成一条（train_faat.py 内部）：
#   (1) 训干净代理 proxy.train_clean_proxy → resource/faat/proxy/*.pth（仅缺失时训）
#   (2) 离线优化 δ_global+π_θ+UnetGen（冻结代理+c_target）→ resource/faat/save_trigger_10_0/
#   (3) 受害模型 train_backdoor.py --backdoor_type faatb → results/faatb_res_square/
# 用法: nohup bash run_faat_stageB.sh >/dev/null 2>&1 &   看进度: tail -f results/_run_faat_stageB.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_stageB.log
FREE_MEM_MB=8000
mkdir -p results

PIDF=results/_run_faat_stageB.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有 Stage B 调度器实例(PID $(cat "$PIDF"))，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== FAAT Stage B 调度器启动（等 Stage A + run_table1 结束后开跑）====" | tee -a "$LOG"

# 1) 等 Stage A 与 blend 都结束
wait_done() {  # $1=pidfile
  while [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; do
    echo "[$(date '+%T')] 等待 $1 (PID $(cat "$1")) 结束..." | tee -a "$LOG"; sleep 300
  done
}
wait_done results/_run_faat_stageA.pid
wait_done results/_run_table1.pid
echo "[$(date '+%F %T')] Stage A 与 run_table1 均已结束，等待空闲 GPU..." | tee -a "$LOG"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v thr="$FREE_MEM_MB" '{gsub(/ /,"",$2); if ($2+0 >= thr) print $1, $2}' | \
  sort -k2 -nr | head -n 1 | awk '{print $1}'
}
GPU=""
while :; do GPU=$(free_gpu); [ -n "$GPU" ] && break; echo "[$(date '+%T')] 无空闲GPU(≥${FREE_MEM_MB}MiB)..." | tee -a "$LOG"; sleep 120; done
echo "[$(date '+%F %T')] 选定 GPU$GPU，启动 Stage B（代理→优化→受害训练）" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

# 2) 一条命令串起三步（train_faat.py: --train_proxy_if_missing 训代理；优化；--train 交 train_backdoor.py）
CUDA_VISIBLE_DEVICES="$GPU" $PY -u train_faat.py \
  --dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --proxy_epochs 100 \
  --save_trigger ./resource/faat/save_trigger_10_0 --steps 2000 --batch_size 48 \
  --init_global_scale 1.0 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_align_global 0.5 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --train --gpu "$GPU" --result_dir results/faatb_res_square \
  >> "$LOG" 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
  echo "[$(date '+%F %T')] ===================== FAAT STAGE B DONE (rc=0) =====================" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] ===================== FAAT STAGE B FAILED (rc=$RC) 见日志 =====================" | tee -a "$LOG"
fi
