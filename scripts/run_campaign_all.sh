#!/bin/bash
# Full multi-dataset campaign: KST (clean-label + Component A) on CIFAR-100/GTSRB/Tiny
# + baseline (Badnets/Blended/MultiBpp-RGB/MultiBpp-B) on CIFAR-100/GTSRB for comparison.
# Tiny baseline already reproduced (results/bl_tiny_*). 5 waves of 4 on GPU 0-3. ~8h.
set -u
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
EP=300; YT=0; SEED=1
KROOT=resource/kst_sdt

run() { gpu=$1; rdir=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_backdoor.py "$@" --y_target $YT --seed $SEED --epochs $EP --result_dir $rdir > $rdir.stdout 2>&1 &
  echo "  [gpu$gpu] $rdir"
}

echo "### CAMPAIGN START $(date) ###"

# --- Wave 1: KST CIFAR-100 (0.5% clean-label) ---
echo "=== W1: KST CIFAR-100 ==="
C="--dataset cifar100 --output_dir ./resource/save_metric_100_res --poison_rate 0.005 --backdoor_type kst"
run 0 results/kst_sdt/c100_kst_e16_forget $C --kst_delta_path $KROOT/kst_delta_cifar100_r4_eps0.0627_a0.0_ce0.pt --selection forget
run 1 results/kst_sdt/c100_kst_e16_reslin $C --kst_delta_path $KROOT/kst_delta_cifar100_r4_eps0.0627_a0.0_ce0.pt --selection res --res_sel linear
run 2 results/kst_sdt/c100_kst_e20_forget $C --kst_delta_path $KROOT/kst_delta_cifar100_r4_eps0.0784_a0.0_ce0.pt --selection forget
run 3 results/kst_sdt/c100_kst_e20_reslin $C --kst_delta_path $KROOT/kst_delta_cifar100_r4_eps0.0784_a0.0_ce0.pt --selection res --res_sel linear
wait; echo "W1 done $(date)"

# --- Wave 2: KST GTSRB (1% clean-label) ---
echo "=== W2: KST GTSRB ==="
G="--dataset gtsrb --data_dir ./data/GTSRB32 --num_classes 43 --output_dir ./resource/save_metric_gtsrb --poison_rate 0.01 --backdoor_type kst"
run 0 results/kst_sdt/gtsrb_kst_e16_forget $G --kst_delta_path $KROOT/kst_delta_gtsrb_r4_eps0.0627_a0.0_ce0.pt --selection forget
run 1 results/kst_sdt/gtsrb_kst_e16_reslin $G --kst_delta_path $KROOT/kst_delta_gtsrb_r4_eps0.0627_a0.0_ce0.pt --selection res --res_sel linear
run 2 results/kst_sdt/gtsrb_kst_e20_forget $G --kst_delta_path $KROOT/kst_delta_gtsrb_r4_eps0.0784_a0.0_ce0.pt --selection forget
run 3 results/kst_sdt/gtsrb_kst_e20_reslin $G --kst_delta_path $KROOT/kst_delta_gtsrb_r4_eps0.0784_a0.0_ce0.pt --selection res --res_sel linear
wait; echo "W2 done $(date)"

# --- Wave 3: baseline CIFAR-100 (res/linear, 0.5%) ---
echo "=== W3: baseline CIFAR-100 ==="
BC="--dataset cifar100 --output_dir ./resource/save_metric_100_res --poison_rate 0.005 --selection res --res_sel linear"
run 0 results/kst_sdt/c100_bl_badnets  $BC --backdoor_type badnets --type 0:0:0
run 1 results/kst_sdt/c100_bl_blend    $BC --backdoor_type blend --type 2:2:2
run 2 results/kst_sdt/c100_bl_mbpprgb  $BC --backdoor_type quantize --num_levels 24:28:8
run 3 results/kst_sdt/c100_bl_mbppB    $BC --backdoor_type quantize --num_levels 255:255:8
wait; echo "W3 done $(date)"

# --- Wave 4: baseline GTSRB (res/linear, 1%) ---
echo "=== W4: baseline GTSRB ==="
BG="--dataset gtsrb --data_dir ./data/GTSRB32 --num_classes 43 --output_dir ./resource/save_metric_gtsrb --poison_rate 0.01 --selection res --res_sel linear"
run 0 results/kst_sdt/gtsrb_bl_badnets  $BG --backdoor_type badnets --type 0:0:0
run 1 results/kst_sdt/gtsrb_bl_blend    $BG --backdoor_type blend --type 2:2:2
run 2 results/kst_sdt/gtsrb_bl_mbpprgb  $BG --backdoor_type quantize --num_levels 24:28:8
run 3 results/kst_sdt/gtsrb_bl_mbppB    $BG --backdoor_type quantize --num_levels 255:255:8
wait; echo "W4 done $(date)"

# --- Wave 5: KST Tiny (0.25% clean-label) ---
echo "=== W5: KST Tiny ==="
T="--dataset tiny --data_dir ./data_tiny --num_classes 200 --output_dir ./resource/save_metric_tiny_res --poison_rate 0.0025 --backdoor_type kst"
run 0 results/kst_sdt/tiny_kst_e16_forget $T --kst_delta_path $KROOT/kst_delta_tiny_r4_eps0.0627_a0.0_ce0.pt --selection forget
run 1 results/kst_sdt/tiny_kst_e16_reslin $T --kst_delta_path $KROOT/kst_delta_tiny_r4_eps0.0627_a0.0_ce0.pt --selection res --res_sel linear
run 2 results/kst_sdt/tiny_kst_e20_forget $T --kst_delta_path $KROOT/kst_delta_tiny_r4_eps0.0784_a0.0_ce0.pt --selection forget
run 3 results/kst_sdt/tiny_kst_e20_reslin $T --kst_delta_path $KROOT/kst_delta_tiny_r4_eps0.0784_a0.0_ce0.pt --selection res --res_sel linear
wait; echo "W5 done $(date)"

echo "### CAMPAIGN ALL DONE $(date) ###"
