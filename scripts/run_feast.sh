#!/bin/bash
# FEAST validation (4-GPU). Decisive test of the FEATURE-STARVATION mechanism.
# User's new angle: starve natural target features (untargeted adversarial push, L_inf 8/255) so a
# weak trigger becomes the only learnable signal -> break the stealth/ASR wall.
#
# Smoke findings going in:
#   * phase trigger alone WALLS (proxyASR 0.06 at huge visible L2=6.1) -- phase carries structure,
#     so it is highly visible AND a weak adversarial direction (Oppenheim-Lim).
#   * starvation WORKS: CE-to-target 10.4->24.6 at L_inf=8/255 (adversarial-budget stealth).
#   * pixel trigger (Narcissus-style) is an effective direction (proxyASR 0.58 @ L2=1.0, L_inf 0.09).
#
# GPU0 vs GPU1 (pixel b1.0, starve vs no-starve) = does starvation help an effective trigger?
# GPU2 (pixel b0.5 + starve)                     = does starvation rescue a WEAKER trigger?
# GPU3 (phase + starve)                           = user's exact design: does starvation rescue phase?
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
mkdir -p logs/feast
EPS=0.03137  # 8/255

CUDA_VISIBLE_DEVICES=0 nohup $PY -u -m faat.train_feast --trigger_mode pixel --budget 1.0 \
  --starve_eps $EPS --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/feast_pix_b1.0_starve_s1 --save_trigger resource/faat/feast/pix_b1.0_starve_s1 \
  > logs/feast/pix_b1.0_starve_s1.log 2>&1 &
echo "GPU0 pixel b1.0 + starve PID=$!"

CUDA_VISIBLE_DEVICES=1 nohup $PY -u -m faat.train_feast --trigger_mode pixel --budget 1.0 \
  --starve_eps 0 --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/feast_pix_b1.0_nostarve_s1 --save_trigger resource/faat/feast/pix_b1.0_nostarve_s1 \
  > logs/feast/pix_b1.0_nostarve_s1.log 2>&1 &
echo "GPU1 pixel b1.0 NO-starve (control) PID=$!"

CUDA_VISIBLE_DEVICES=2 nohup $PY -u -m faat.train_feast --trigger_mode pixel --budget 0.5 \
  --starve_eps $EPS --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/feast_pix_b0.5_starve_s1 --save_trigger resource/faat/feast/pix_b0.5_starve_s1 \
  > logs/feast/pix_b0.5_starve_s1.log 2>&1 &
echo "GPU2 pixel b0.5 + starve PID=$!"

CUDA_VISIBLE_DEVICES=3 nohup $PY -u -m faat.train_feast --trigger_mode phase --phi_max 0.6 --radius 8 \
  --starve_eps $EPS --y_target 0 --seed 1 --steps 3000 \
  --result_dir results/feast_phase_starve_s1 --save_trigger resource/faat/feast/phase_starve_s1 \
  > logs/feast/phase_starve_s1.log 2>&1 &
echo "GPU3 phase + starve (user's design) PID=$!"

wait
echo "=== FEAST VALIDATION DONE ==="
