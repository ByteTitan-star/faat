#!/bin/bash
# PA-ICT 4-GPU sweep: does PCA-subspace alignment (lambda>0) improve AC/SS evasion over
# CE-only (lambda=0 = ICIT), at matched ASR/stealth? budget=2.0 fixed; vary lambda + objective.
# GPU0 lambda=1.0 CE (main) | GPU1 lambda=0.1 CE | GPU2 lambda=0 CE (ICIT baseline) | GPU3 lambda=1.0 no-CE (PCA-only)
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/pca

# GPU0: lambda=1.0, CE  (main proposal)
CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 1.0 --k 256 \
  --y_target 0 --seed 1 --steps 3000 --result_dir results/pca_b2.0_l1.0_ce_s1 \
  --save_trigger resource/faat/pca/b2.0_l1.0_ce > logs/pca/g0_l1.0_ce.log 2>&1 &
echo "GPU0 lambda=1.0 CE PID=$!"

# GPU1: lambda=0.1, CE  (less PCA weight)
CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 0.1 --k 256 \
  --y_target 0 --seed 1 --steps 3000 --result_dir results/pca_b2.0_l0.1_ce_s1 \
  --save_trigger resource/faat/pca/b2.0_l0.1_ce > logs/pca/g1_l0.1_ce.log 2>&1 &
echo "GPU1 lambda=0.1 CE PID=$!"

# GPU2: lambda=0.0, CE  (CE-only = ICIT single-proxy baseline)
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 0.0 --k 256 \
  --y_target 0 --seed 1 --steps 3000 --result_dir results/pca_b2.0_l0.0_ce_s1 \
  --save_trigger resource/faat/pca/b2.0_l0.0_ce > logs/pca/g2_l0.0_ce.log 2>&1 &
echo "GPU2 lambda=0.0 CE (ICIT) PID=$!"

# GPU3: lambda=1.0, no-CE  (PCA-align only -- does alignment alone flip the proxy?)
CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m faat.train_pca --budget 2.0 --lambda_pca 1.0 --k 256 --no_ce \
  --y_target 0 --seed 1 --steps 3000 --result_dir results/pca_b2.0_l1.0_noce_s1 \
  --save_trigger resource/faat/pca/b2.0_l1.0_noce > logs/pca/g3_l1.0_noce.log 2>&1 &
echo "GPU3 lambda=1.0 no-CE PID=$!"

wait
echo "=== ALL 4 PA-ICT RUNS DONE ==="
