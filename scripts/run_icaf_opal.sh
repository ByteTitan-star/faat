#!/usr/bin/env bash
# Launch ICAF (isophote chromatic-aberration warp) + OPAL (ordinal polytope) victim training on
# 4 GPUs (CIFAR-10, 1% poison, R18, Res-x^2 selection, 300ep, baseline schedule).
# PA-ICT skipped (prior art: SGBA Luo et al. ASOC 2025). All 4 GPUs -> new schemes.
set -u
cd "$(dirname "$0")/.."
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/icaf logs/opal

# ---- ICAF: alpha sweep (proxy says wall; confirm victim ASR + SSIM) ----
CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_icaf \
  --alpha 1.0 --mask_res 8 --steps 3000 --seed 1 \
  --save_trigger resource/faat/icaf/a1.0_s1 \
  --result_dir results/icaf_a1.0_s1 > logs/icaf/a1.0_s1.stdout 2>&1 &
echo $! > logs/icaf/a1.0_s1.pid

CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_icaf \
  --alpha 2.0 --mask_res 8 --steps 3000 --seed 1 \
  --save_trigger resource/faat/icaf/a2.0_s1 \
  --result_dir results/icaf_a2.0_s1 > logs/icaf/a2.0_s1.stdout 2>&1 &
echo $! > logs/icaf/a2.0_s1.pid

# ---- OPAL: bounded binary ordinal key, block granularity sweep ----
# margin=8/255, linf=16/255 (Narcissus-level stealth), binary levels, M=128 pairs
CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m faat.train_opal \
  --block 4 --M 128 --margin 0.03137 --linf 0.0627 --levels binary --key_seed 1 --seed 1 \
  --save_trigger resource/faat/opal/b4_m8_l16_s1 \
  --result_dir results/opal_b4_m8_l16_s1 > logs/opal/b4_m8_l16_s1.stdout 2>&1 &
echo $! > logs/opal/b4_m8_l16_s1.pid

CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m faat.train_opal \
  --block 8 --M 128 --margin 0.03137 --linf 0.0627 --levels binary --key_seed 1 --seed 1 \
  --save_trigger resource/faat/opal/b8_m8_l16_s1 \
  --result_dir results/opal_b8_m8_l16_s1 > logs/opal/b8_m8_l16_s1.stdout 2>&1 &
echo $! > logs/opal/b8_m8_l16_s1.pid

echo "launched 4 runs:"; cat logs/icaf/*.pid logs/opal/*.pid
