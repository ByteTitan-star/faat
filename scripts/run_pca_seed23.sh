#!/bin/bash
# Multi-seed confirmation (seeds 2,3) of the PA-ICT headline: pure-PCA (no-CE) evades NC vs
# ICIT (lambda=0 CE) caught. 4 jobs: noce s2/s3 + l0.0_ce s2/s3 on GPU0-3.
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pca

CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 1.0 --k 256 --no_ce \
  --y_target 0 --seed 2 --steps 3000 --result_dir results/pca_b2.0_l1.0_noce_s2 \
  --save_trigger resource/faat/pca/b2.0_l1.0_noce_s2 > logs/pca/noce_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 1.0 --k 256 --no_ce \
  --y_target 0 --seed 3 --steps 3000 --result_dir results/pca_b2.0_l1.0_noce_s3 \
  --save_trigger resource/faat/pca/b2.0_l1.0_noce_s3 > logs/pca/noce_s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 0.0 --k 256 \
  --y_target 0 --seed 2 --steps 3000 --result_dir results/pca_b2.0_l0.0_ce_s2 \
  --save_trigger resource/faat/pca/b2.0_l0.0_ce_s2 > logs/pca/icit_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 0.0 --k 256 \
  --y_target 0 --seed 3 --steps 3000 --result_dir results/pca_b2.0_l0.0_ce_s3 \
  --save_trigger resource/faat/pca/b2.0_l0.0_ce_s3 > logs/pca/icit_s3.log 2>&1 &
echo "launched 4 seed-confirmation runs"
wait
echo "=== seed confirmation DONE ==="