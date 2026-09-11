#!/bin/bash
# Phase 1: ensemble-ICIT end-to-end + black-box transfer to held-out architectures.
# GPU1: ensemble-ICIT (R18+34+50 surrogates) -> R18 victim (ASR/BA/AC/SS).
# GPU2: clean ResNet101 victim (held-out transfer target).
# GPU3: clean ResNet152 victim (held-out transfer target, deeper).
# When all 3 finish -> run black-box transfer eval (ensemble vs single-R18 on R101/R152).
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=logs/phase1.log
mkdir -p logs/phase1
echo "[$(date '+%F %T')] === Phase 1 启动: 集成ICIT-R18 + 干净R101/R152 (held-out) ===" | tee -a $LOG

PROXIES=resource/faat/proxy/resnet18_clean_cifar10.pth,resource/faat/proxy/resnet34_clean_cifar10.pth,resource/faat/proxy/resnet50_clean_cifar10.pth
ARCHES=resnet18,resnet34,resnet50

# 1. ensemble-ICIT R18 victim (GPU1)
CUDA_VISIBLE_DEVICES=1 nohup $PY -m faat.train_icit_ens --dataset cifar10 --data_dir ./data \
   --num_classes 10 --proxies $PROXIES --arches $ARCHES --victim_arch resnet18 \
   --save_trigger resource/faat/icit_ens/cifar10_b2.0_s1 --budget 2.0 --poison_rate 0.01 \
   --y_target 0 --seed 1 --epochs 300 --steps 3000 --output_dir ./resource/save_metric_10_res \
   --result_dir results/icit_ens_cifar10_b2.0_s1 --gpu 1 > logs/phase1/icit_ens_r18.log 2>&1 &
ENS=$!
echo "[$(date '+%F %T')] ensemble-ICIT R18 victim PID=$ENS GPU1" | tee -a $LOG

# 2. clean ResNet101 (GPU2, held-out)
CUDA_VISIBLE_DEVICES=2 nohup $PY -m faat._train_proxy_alt --arch resnet101 --dataset cifar10 \
   --num_classes 10 --epochs 100 --save_path resource/faat/proxy/resnet101_clean_cifar10.pth --seed 1 \
   > logs/phase1/proxy_resnet101.log 2>&1 &
R101=$!
echo "[$(date '+%F %T')] clean ResNet101 PID=$R101 GPU2" | tee -a $LOG

# 3. clean ResNet152 (GPU3, held-out, deeper)
CUDA_VISIBLE_DEVICES=3 nohup $PY -m faat._train_proxy_alt --arch resnet152 --dataset cifar10 \
   --num_classes 10 --epochs 100 --save_path resource/faat/proxy/resnet152_clean_cifar10.pth --seed 1 \
   > logs/phase1/proxy_resnet152.log 2>&1 &
R152=$!
echo "[$(date '+%F %T')] clean ResNet152 PID=$R152 GPU3" | tee -a $LOG

echo "[$(date '+%F %T')] 等待 3 个任务完成..." | tee -a $LOG
wait $ENS; echo "[$(date '+%F %T')] ensemble-ICIT R18 victim DONE" | tee -a $LOG
wait $R101; echo "[$(date '+%F %T')] ResNet101 DONE" | tee -a $LOG
wait $R152; echo "[$(date '+%F %T')] ResNet152 DONE" | tee -a $LOG

# eval: ASR/BA on R18 victim
echo "[$(date '+%F %T')] === 黑盒迁移评估 (集成 vs 单R18, held-out R101/R152) ===" | tee -a $LOG
$PY -m faat._icit_blackbox_transfer 2>&1 | tee -a $LOG
# AC/SS on the ensemble R18 victim
$PY -m faat._icit_defenses --rdir results/icit_ens_cifar10_b2.0_s1 \
   --gen resource/faat/icit_ens/cifar10_b2.0_s1 --dataset cifar10 --data_dir ./data 2>&1 | tee -a $LOG
echo "[$(date '+%F %T')] === Phase 1 ALL DONE ===" | tee -a $LOG
