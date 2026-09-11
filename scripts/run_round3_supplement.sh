#!/bin/bash
# =============================================================================
# Round-3 supplement — multi-seed robustness for the KST chapter (Direction A)
# =============================================================================
# Round-1 (campaign) + Round-2 found these STRONG KST results but each is seed1-only.
# Round-3 adds seed2/seed3 so every headline number has variance bars, plus completes
# the GTSRB KST epsilon axis (e16/e20 already ASR=0; add e48 to seal the limitation).
#
# 10 runs across 4 GPUs (all 4 free as of 2026-08-05; wanaihua/adapgc finished):
#   Tiny ε48     forget/reslin × seed{2,3}   = 4 runs   (~3.0h each)
#   CIFAR-100 ε48 forget/reslin × seed{2,3}  = 4 runs   (~1.4h each)
#   GTSRB KST ε48 forget/reslin × seed1      = 2 runs   (~1.0h each)
#
# HARD RULES: NEW dirs only (_s2/_s3); idempotent on output_{seed}.log ep299; GPU3 OK now.
# Provenance: HEAD 4097c94 (exp/kst-sdt). See docs/round2_supplement.md (Round-3 section).
# =============================================================================
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
EP=300; YT=0

# completion = epoch 299 row in this seed's own log file
is_done() { local log="$1/output_$2.log"; [ -f "$log" ] && grep -qE '\] - 299[[:space:]]' "$log"; }

# run <gpu> <result_dir> <seed> -- <train_backdoor.py args...>
run() {
  local gpu=$1 rdir=$2 seed=$3; shift 3
  if is_done "$rdir" "$seed"; then echo "[SKIP $(date +%H:%M)] $rdir s$seed (ep299 reached)"; return 0; fi
  mkdir -p "$rdir"
  echo "[START $(date '+%m-%d %H:%M')][gpu$gpu] seed=$seed -> $rdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_backdoor.py "$@" \
      --y_target $YT --seed $seed --epochs $EP --result_dir "$rdir" > "$rdir.stdout" 2>&1
  local rc=$?
  echo "[DONE  $(date '+%m-%d %H:%M')] $rdir s$seed exit=$rc"
}

# ---- config shortcuts ----
TINY48="--dataset tiny   --data_dir ./data_tiny --num_classes 200 --output_dir ./resource/save_metric_tiny_res --poison_rate 0.0025 --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.1882_a0.0_ce0.pt"
C10048="--dataset cifar100                       --output_dir ./resource/save_metric_100_res  --poison_rate 0.005  --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_cifar100_r4_eps0.1882_a0.0_ce0.pt"
GT48="--dataset gtsrb    --data_dir ./data/GTSRB32 --num_classes 43  --output_dir ./resource/save_metric_gtsrb --poison_rate 0.01  --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_gtsrb_r4_eps0.1882_a0.0_ce0.pt"

gpu0_queue() {  # ~5.4h
  run 0 results/kst_sdt/tiny_kst_e48_forget_s2 2  $TINY48 --selection forget
  run 0 results/kst_sdt/c100_kst_e48_forget_s2  2  $C10048 --selection forget
  run 0 results/kst_sdt/gtsrb_kst_e48_forget    1  $GT48   --selection forget
}
gpu1_queue() {  # ~5.4h
  run 1 results/kst_sdt/tiny_kst_e48_forget_s3 3  $TINY48 --selection forget
  run 1 results/kst_sdt/c100_kst_e48_forget_s3  3  $C10048 --selection forget
  run 1 results/kst_sdt/gtsrb_kst_e48_reslin    1  $GT48   --selection res --res_sel linear
}
gpu2_queue() {  # ~4.4h
  run 2 results/kst_sdt/tiny_kst_e48_reslin_s2 2  $TINY48 --selection res --res_sel linear
  run 2 results/kst_sdt/c100_kst_e48_reslin_s2 2  $C10048 --selection res --res_sel linear
}
gpu3_queue() {  # ~4.4h
  run 3 results/kst_sdt/tiny_kst_e48_reslin_s3 3  $TINY48 --selection res --res_sel linear
  run 3 results/kst_sdt/c100_kst_e48_reslin_s3 3  $C10048 --selection res --res_sel linear
}

case "${1:-all}" in
  gpu0) gpu0_queue ;;
  gpu1) gpu1_queue ;;
  gpu2) gpu2_queue ;;
  gpu3) gpu3_queue ;;
  all)
    echo "### ROUND-3 SUPPLEMENT START $(date) ### [4 GPUs, 10 runs]"
    gpu0_queue & P0=$!
    gpu1_queue & P1=$!
    gpu2_queue & P2=$!
    gpu3_queue & P3=$!
    echo "queue pids: gpu0=$P0 gpu1=$P1 gpu2=$P2 gpu3=$P3"
    wait $P0 $P1 $P2 $P3
    echo "### ROUND-3 SUPPLEMENT ALL DONE $(date) ###"
    ;;
  *) echo "usage: $0 [all|gpu0|gpu1|gpu2|gpu3]"; exit 2 ;;
esac
