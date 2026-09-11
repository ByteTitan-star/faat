#!/bin/bash
# =============================================================================
# Round-2 supplement experiments — Direction A (FAAT main axis + KST section)
# =============================================================================
# Fills the verified gaps from campaign audit (2026-08-04):
#   1. GTSRB baseline full 300ep  — campaign W4 was KILLED @ ~ep24 (results/kst_sdt/gtsrb_bl_* truncated)
#   2. Tiny-ImageNet KST full 300ep — NEVER run (only diag_* short probes exist; no tiny_kst_*/output_1.log)
#   3. CIFAR-100 KST ε48 full — ALREADY DONE (c100_kst_e48_* reached ep299); NOT re-run, just re-parsed
#   4. CIFAR-10 KST multi-seed — only seed1 exists (cl_kst_*); add seed2/seed3
#
# HARD RULES (per user):
#   * NEVER overwrite existing results — all outputs go to NEW dirs (_full / _s2 / _s3 / tiny_kst_e48_*).
#   * Idempotent: a run whose output_1.log already reached epoch 299 is SKIPPED.
#   * GPUs 0/1/2 only — GPU3 is in use by another user (wanaihua / adapgc), do NOT touch.
#
# Audit baseline HEAD: 4097c94 (exp/kst-sdt). See docs/round2_supplement.md for full provenance.
# =============================================================================
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
EP=300; YT=0

# completion = epoch 299 row present in the per-run log
is_done() { local log="$1/output_1.log"; [ -f "$log" ] && grep -qE '\] - 299[[:space:]]' "$log"; }

# run <gpu> <result_dir> <seed> -- <train_backdoor.py args...>
run() {
  local gpu=$1 rdir=$2 seed=$3; shift 3
  if is_done "$rdir"; then echo "[SKIP $(date +%H:%M)] $rdir (ep299 reached)"; return 0; fi
  mkdir -p "$rdir"
  echo "[START $(date '+%m-%d %H:%M')][gpu$gpu] seed=$seed -> $rdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_backdoor.py "$@" \
      --y_target $YT --seed $seed --epochs $EP --result_dir "$rdir" > "$rdir.stdout" 2>&1
  local rc=$?
  echo "[DONE  $(date '+%m-%d %H:%M')] $rdir exit=$rc"
}

# ---- config shortcuts ----
C10="--dataset cifar10 --output_dir ./resource/save_metric_10_res --poison_rate 0.01 --backdoor_type kst"
TINY="--dataset tiny --data_dir ./data_tiny --num_classes 200 --output_dir ./resource/save_metric_tiny_res --poison_rate 0.0025 --backdoor_type kst"
GTBL="--dataset gtsrb --data_dir ./data/GTSRB32 --num_classes 43 --output_dir ./resource/save_metric_gtsrb --poison_rate 0.01 --selection res --res_sel linear"

# ---- GPU0 queue (~6.4h): 1 Tiny + 2 CIFAR10-seed + 1 GTSRB ----
gpu0_queue() {
  run 0 results/kst_sdt/tiny_kst_e48_forget      1  $TINY --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.1882_a0.0_ce0.pt --selection forget
  run 0 results/kst_sdt/cl_kst_e16_forget_s2     2  $C10 --kst_delta_path ./resource/kst_sdt/kst_delta_r4_eps0.0627.pt         --selection forget
  run 0 results/kst_sdt/cl_kst_e20_reslinear_s2  2  $C10 --kst_delta_path ./resource/kst_sdt/kst_delta_r4_eps0.0784.pt         --selection res --res_sel linear
  run 0 results/kst_sdt/gtsrb_bl_badnets_full    1  $GTBL --backdoor_type badnets  --type 0:0:0
}

# ---- GPU1 queue (~6.4h): 1 Tiny + 2 CIFAR10-seed + 1 GTSRB ----
gpu1_queue() {
  run 1 results/kst_sdt/tiny_kst_e48_reslin       1  $TINY --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.1882_a0.0_ce0.pt --selection res --res_sel linear
  run 1 results/kst_sdt/cl_kst_e16_forget_s3      3  $C10 --kst_delta_path ./resource/kst_sdt/kst_delta_r4_eps0.0627.pt         --selection forget
  run 1 results/kst_sdt/cl_kst_e20_reslinear_s3   3  $C10 --kst_delta_path ./resource/kst_sdt/kst_delta_r4_eps0.0784.pt         --selection res --res_sel linear
  run 1 results/kst_sdt/gtsrb_bl_blend_full       1  $GTBL --backdoor_type blend    --type 2:2:2
}

# ---- GPU2 queue (~7h): 2 Tiny + 2 GTSRB ----
gpu2_queue() {
  run 2 results/kst_sdt/tiny_kst_e16_forget      1  $TINY --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.0627_a0.0_ce0.pt --selection forget
  run 2 results/kst_sdt/tiny_kst_e16_reslin      1  $TINY --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.0627_a0.0_ce0.pt --selection res --res_sel linear
  run 2 results/kst_sdt/gtsrb_bl_mbpprgb_full    1  $GTBL --backdoor_type quantize --num_levels 24:28:8
  run 2 results/kst_sdt/gtsrb_bl_mbppB_full      1  $GTBL --backdoor_type quantize --num_levels 255:255:8
}

case "${1:-all}" in
  gpu0) gpu0_queue ;;
  gpu1) gpu1_queue ;;
  gpu2) gpu2_queue ;;
  all)
    echo "### ROUND-2 SUPPLEMENT START $(date) ###"
    gpu0_queue & P0=$!
    gpu1_queue & P1=$!
    gpu2_queue & P2=$!
    echo "queue pids: gpu0=$P0 gpu1=$P1 gpu2=$P2"
    wait $P0 $P1 $P2
    echo "### ROUND-2 SUPPLEMENT ALL DONE $(date) ###"
    ;;
  *) echo "usage: $0 [all|gpu0|gpu1|gpu2]"; exit 2 ;;
esac
