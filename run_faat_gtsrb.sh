#!/bin/bash
# run_faat_gtsrb.sh — GTSRB 跨数据集 v3.1 验证。
# 等 cal_metric GTSRB 的 epoch_10 pkl + 空闲(≥8GB)GPU，然后跑 2 个 L2 档(1.3隐蔽 / 3.0强ASR)。
# GTSRB 无现成 Narcissus → --global_obj asr --init_random 自生成 δ_global(CE骗代理)，+ 有界adaptive。
# 每档 = 训GTSRB干净代理(43类) → 离线优化 → 受害训练(用 --train 交接)。
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_faat_gtsrb.log
FREE_MEM_MB=8000
mkdir -p results
PIDF=results/_run_faat_gtsrb.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then echo "已有实例，退出"|tee -a "$LOG"; exit 0; fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT
echo "[$(date '+%F %T')] ==== GTSRB v3.1 调度器启动(等pkl+空卡) ====" | tee -a "$LOG"

# 等 GTSRB 选择 pkl
PKL=resource/save_metric_gtsrb/resnet_loss_grad_epoch_10_seed_1.pkl
while [ ! -f "$PKL" ]; do echo "[$(date '+%T')] 等 GTSRB pkl..."|tee -a "$LOG"; sleep 60; done
echo "[$(date '+%T')] GTSRB pkl 就绪"|tee -a "$LOG"

free_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v t="$FREE_MEM_MB" '$1==3 {gsub(/ /,"",$2); if($2+0>=t) print 3}' | head -1; }

COMMON="--dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 --y_target 0 --selection res --res_sel square \
  --poison_rate 0.01 --output_dir ./resource/save_metric_gtsrb --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --proxy_path ./resource/faat/proxy/resnet18_clean_gtsrb.pth \
  --steps 2000 --batch_size 48 --global_obj asr --init_random --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 --train"

for L2 in 1.3 3.0; do
  while :; do GPU=$(free_gpu); [ -n "$GPU" ] && break; echo "[$(date '+%T')] 无空卡，等GTSRB L2=$L2..."|tee -a "$LOG"; sleep 90; done
  ST="./resource/faat/gtsrb/l2_${L2}"; RD="results/faatb_gtsrb_l2_${L2}"; mkdir -p "$RD"
  echo "[$(date '+%T')] 启动 GTSRB L2=$L2 on GPU$GPU -> $RD"|tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_faat.py $COMMON --global_l2_max "$L2" \
    --save_trigger "$ST" --gpu "$GPU" --result_dir "$RD" > "${RD}.stdout" 2>&1 &
  sleep 20
done
wait
echo "[$(date '+%F %T')] ===================== FAAT GTSRB DONE =====================" | tee -a "$LOG"
