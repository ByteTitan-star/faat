#!/bin/bash
# =============================================================================
# Round-3b: recover the 3 GPU0 runs that OOM'd at round-3 launch.
# =============================================================================
# At round-3 launch (10:34) GPU0 looked free, but wanaihua started a "Factory" job on
# GPU0 seconds later, so the 3 GPU0 runs (tiny_forget_s2 / c100_forget_s2 / gtsrb_forget)
# hit CUDA OOM at model.cuda() and exited=1. GPU1/2/3 are fine.
#
# This script WAITS for round-3 (babgwxqy6) to finish, then runs the 3 lost runs on
# GPU1/2/3 (GPU0 may still be occupied by wanaihua). Idempotent on output_{seed}.log.
# =============================================================================
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
EP=300; YT=0

is_done() { local log="$1/output_$2.log"; [ -f "$log" ] && grep -qE '\] - 299[[:space:]]' "$log"; }
run() {
  local gpu=$1 rdir=$2 seed=$3; shift 3
  if is_done "$rdir" "$seed"; then echo "[SKIP $(date +%H:%M)] $rdir s$seed (ep299)"; return 0; fi
  mkdir -p "$rdir"
  echo "[START $(date '+%m-%d %H:%M')][gpu$gpu] seed=$seed -> $rdir"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_backdoor.py "$@" \
      --y_target $YT --seed $seed --epochs $EP --result_dir "$rdir" > "$rdir.stdout" 2>&1
  echo "[DONE  $(date '+%m-%d %H:%M')] $rdir s$seed exit=$?"
}

TINY48="--dataset tiny --data_dir ./data_tiny --num_classes 200 --output_dir ./resource/save_metric_tiny_res --poison_rate 0.0025 --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.1882_a0.0_ce0.pt"
C10048="--dataset cifar100 --output_dir ./resource/save_metric_100_res --poison_rate 0.005 --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_cifar100_r4_eps0.1882_a0.0_ce0.pt"
GT48="--dataset gtsrb --data_dir ./data/GTSRB32 --num_classes 43 --output_dir ./resource/save_metric_gtsrb --poison_rate 0.01 --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_gtsrb_r4_eps0.1882_a0.0_ce0.pt"

echo "### ROUND-3B: waiting for round-3 (run_round3_supplement.sh) to finish... $(date) ###"
while pgrep -f "run_round3_supplement.sh" >/dev/null 2>&1; do sleep 120; done
echo "### ROUND-3B START $(date) ### (round-3 done; using GPU1/2/3, avoiding GPU0)"

# 3 recovered runs, one per free GPU (concurrent)
run 1 results/kst_sdt/tiny_kst_e48_forget_s2 2 $TINY48 --selection forget &
P1=$!
run 2 results/kst_sdt/c100_kst_e48_forget_s2  2 $C10048 --selection forget &
P2=$!
run 3 results/kst_sdt/gtsrb_kst_e48_forget    1 $GT48   --selection forget &
P3=$!
wait $P1 $P2 $P3
echo "### ROUND-3B ALL DONE $(date) ###"
