#!/bin/bash
# run_cifar100_chain.sh — 自主串联 CIFAR-100 三波实验 (v6 → v6b → v6c), GPU1 全程排除。
# v6  (FAAT 1%,    9 组) — 当前已在跑 (brqw3szx3), 本脚本等它跑完。
# v6b (baselines 1%, 9 组) — 方案B matched 对比。
# v6c (FAAT 0.5%,  9 组) — 方案A 直比论文 Table2。
# 每波之间写里程碑到 logs/cifar100_chain.log。Recording(eval_all) 由主控在各里程碑做,本脚本只调度。
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
CHAIN_LOG=logs/cifar100_chain.log
mkdir -p logs

n_done() {  # arg: state json path -> print count of status==done
  $PY -c "import json,sys; s=json.load(open('$1')); print(sum(1 for v in s.values() if v.get('status')=='done'))" 2>/dev/null
}

echo "[$(date '+%F %T')] === CIFAR-100 chain 启动 (GPU1 全程排除) ===" | tee -a $CHAIN_LOG

# --- 等 v6 (1% FAAT) 跑完 9 组 ---
echo "[$(date '+%F %T')] 等待 v6 (1% FAAT) 完成 9/9 ..." | tee -a $CHAIN_LOG
while [ "$(n_done logs/v6_cifar100/scheduler_state.json)" != "9" ]; do sleep 60; done
echo "[$(date '+%F %T')] v6 DONE (9/9). 启动 v6b (baselines @1%, 方案B) ..." | tee -a $CHAIN_LOG

# --- v6b: baselines @1% ---
$PY -m faat.scheduler --queue logs/v6b_cifar100_baselines/queue.json \
   --log_dir logs/v6b_cifar100_baselines --state logs/v6b_cifar100_baselines/scheduler_state.json \
   --exclude 1 --poll 20 >> $CHAIN_LOG 2>&1
echo "[$(date '+%F %T')] v6b DONE (9/9 baselines). 启动 v6c (FAAT @0.5%, 方案A) ..." | tee -a $CHAIN_LOG

# --- v6c: FAAT @0.5% ---
$PY -m faat.scheduler --queue logs/v6c_cifar100_faat05/queue.json \
   --log_dir logs/v6c_cifar100_faat05 --state logs/v6c_cifar100_faat05/scheduler_state.json \
   --exclude 1 --poll 20 >> $CHAIN_LOG 2>&1
echo "[$(date '+%F %T')] === chain ALL DONE: v6 + v6b + v6c (27 runs) ===" | tee -a $CHAIN_LOG
