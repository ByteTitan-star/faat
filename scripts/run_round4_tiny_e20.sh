#!/bin/bash
# =============================================================================
# Round-4: fill the Tiny ε20 full-300ep gap.
# =============================================================================
# Tiny KST had e16 and e48 full runs but ε20 was never run (only diag_tiny_e20 short
# probe). This completes the Tiny ε-Pareto (e16 / e20 / e48). 2 runs, seed1.
#
# Scheduling (maximize GPU use, no idle):
#   tiny_e20_forget  -> GPU3 immediately (GPU3 is free now)
#   tiny_e20_reslin  -> GPU2 as soon as c100_kst_e48_forget_s2 (round-3b) hits ep299
# Idempotent on output_{seed}.log ep299.
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

TINYE20="--dataset tiny --data_dir ./data_tiny --num_classes 200 --output_dir ./resource/save_metric_tiny_res --poison_rate 0.0025 --backdoor_type kst --kst_delta_path ./resource/kst_sdt/kst_delta_tiny_r4_eps0.0784_a0.0_ce0.pt"

echo "### ROUND-4 START $(date) ### [fill Tiny ε20 gap]"
# GPU3: forget now
run 3 results/kst_sdt/tiny_kst_e20_forget 1 $TINYE20 --selection forget &
P1=$!
# GPU2: reslin after round-3b's c100_forget_s2 finishes (frees GPU2)
echo "[round-4] waiting for GPU2 (c100_kst_e48_forget_s2 to hit ep299)..."
while ! is_done results/kst_sdt/c100_kst_e48_forget_s2 2; do sleep 120; done
echo "[round-4] GPU2 free $(date) -> tiny_e20_reslin"
run 2 results/kst_sdt/tiny_kst_e20_reslin 1 $TINYE20 --selection res --res_sel linear &
P2=$!
wait $P1 $P2
echo "### ROUND-4 ALL DONE $(date) ###"
