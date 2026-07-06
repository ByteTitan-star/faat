#!/bin/bash
# run_table1.sh — 后台跑完 Table 1 全部 4 列（Badnets-C / Blended-C[utils.py 已补丁0.2:0.2:0.2] / MultiBpp-B / MultiBpp-RGB[已完成会自动跳过]）
#
# 调度策略：
#   1) 验证组(Res-x² + Forget)优先，便于尽早核对复现趋势；
#   2) 只在「空闲显存 ≥ 8GB」的 GPU 上启动（不抢占他人），并挑空闲最大的卡以自然分散；
#   3) 跳过已跑满 300 epoch 的组、跳过正在运行的组；
#   4) PID 文件单实例保护；全部跑完自动 parse_detail.py --write-md 更新 result_all.md。
#
# 用法：  nohup bash run_table1.sh > /dev/null 2>&1 &     （已脱离会话，可关终端）
#   看进度：tail -f results/_run_table1.log
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main || exit 1
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
LOG=results/_run_table1.log
COMMON="--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10"
MAX_JOBS=4
FREE_MEM_MB=8000   # GPU 空闲显存阈值(MiB)：只占用明显未饱和的卡，避免抢占他人致 OOM
# [复现补丁 2026-07-01] 可选 GPU 黑名单（逗号/空格分隔的 index），用于跳过他人/FAAT 正在用的卡。
# 默认空=不排除（行为同原版）；本次启动用 EXCLUDE_GPU=2 跳过 FAAT 占用的 GPU2。可逆，不影响其它逻辑。
EXCLUDE_GPU="${EXCLUDE_GPU:-}"

mkdir -p results
# 单实例保护：用 PID 文件（避免 flock——子进程会继承 fd 导致锁不释放）。已运行则退出。
PIDF=results/_run_table1.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%F %T')] 已有调度器实例(PID $(cat "$PIDF"))在运行，退出" | tee -a "$LOG"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

echo "[$(date '+%F %T')] ==== Table 1 调度器启动 (4 列: badnets + blend[补丁0.2:0.2:0.2] + quantizeB + quantize) ====" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"

# 任务列表：每行 "result_dir|backdoor_type|extra_args|selection_args"。顺序=优先级。
read -r -d '' JOBS <<'EOF'
badnets_res_square|badnets|--type 0:0:0|--selection res --res_sel square
badnets_forget|badnets|--type 0:0:0|--selection forget
quantizeB_res_square|quantize|--num_levels 255:255:8|--selection res --res_sel square
quantizeB_forget|quantize|--num_levels 255:255:8|--selection forget
badnets_res_linear|badnets|--type 0:0:0|--selection res --res_sel linear
badnets_res_log|badnets|--type 0:0:0|--selection res --res_sel log
badnets_res_exp|badnets|--type 0:0:0|--selection res --res_sel exp
badnets_loss|badnets|--type 0:0:0|--selection loss
badnets_gradient|badnets|--type 0:0:0|--selection grad
badnets_random|badnets|--type 0:0:0|--selection random
quantizeB_res_linear|quantize|--num_levels 255:255:8|--selection res --res_sel linear
quantizeB_res_log|quantize|--num_levels 255:255:8|--selection res --res_sel log
quantizeB_res_exp|quantize|--num_levels 255:255:8|--selection res --res_sel exp
quantizeB_loss|quantize|--num_levels 255:255:8|--selection loss
quantizeB_gradient|quantize|--num_levels 255:255:8|--selection grad
quantizeB_random|quantize|--num_levels 255:255:8|--selection random
blend_res_square|blend|--type 2:2:2|--selection res --res_sel square
blend_forget|blend|--type 2:2:2|--selection forget
blend_res_linear|blend|--type 2:2:2|--selection res --res_sel linear
blend_res_log|blend|--type 2:2:2|--selection res --res_sel log
blend_res_exp|blend|--type 2:2:2|--selection res --res_sel exp
blend_loss|blend|--type 2:2:2|--selection loss
blend_gradient|blend|--type 2:2:2|--selection grad
blend_random|blend|--type 2:2:2|--selection random
EOF

running() { ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -oE -- '--result_dir results/[^ ]+' | sort -u | wc -l; }
is_running() { ps -eo cmd | grep 'train_backdoor.py' | grep -v grep | grep -qF -- "--result_dir results/$1"; }
is_done() { [ -f "results/$1/output_1.log" ] && grep -qE '\] - 299[[:space:]]' "results/$1/output_1.log"; }
free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v thr="$FREE_MEM_MB" -v ex="$EXCLUDE_GPU" '
    BEGIN { n=split(ex, a, /[[:space:],]+/); for (i=1;i<=n;i++) if (a[i]!="") skip[a[i]]=1 }
    { gsub(/ /,"",$1); gsub(/ /,"",$2); if (!($1 in skip) && ($2+0 >= thr)) print $1, $2 }' | \
  sort -k2 -nr | head -n 1 | awk '{print $1}'
}

echo "$JOBS" | grep -vE '^[[:space:]]*#' | while IFS='|' read -r dir atk extra selargs; do
  [ -z "$dir" ] && continue
  if is_done "$dir"; then echo "[$(date '+%T')] 跳过(已完成): $dir" | tee -a "$LOG"; continue; fi
  if is_running "$dir"; then echo "[$(date '+%T')] 跳过(运行中): $dir" | tee -a "$LOG"; continue; fi
  # 等并发槽
  while [ "$(running)" -ge "$MAX_JOBS" ]; do sleep 60; done
  # 等空闲 GPU
  while :; do
    GPU=$(free_gpu)
    [ -n "$GPU" ] && break
    echo "[$(date '+%T')] 无空闲GPU(≥${FREE_MEM_MB}MiB)，等待 $dir..." | tee -a "$LOG"
    sleep 90
  done
  mkdir -p "results/$dir"
  echo "[$(date '+%T')] 启动 $dir on GPU$GPU  [$atk $extra $selargs]" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" nohup $PY -u train_backdoor.py $COMMON --backdoor_type "$atk" $extra $selargs \
      --result_dir "results/$dir" > "results/$dir.stdout" 2>&1 &
  sleep 20   # 错峰，便于 free_gpu 选到不同卡
done

# 等全部训练进程结束
while [ "$(running)" -gt 0 ]; do sleep 120; done
echo "[$(date '+%F %T')] 全部训练结束，更新 result_all.md" | tee -a "$LOG"
$PY parse_detail.py ./results results/_detail_summary.json --write-md docs/result_all.md 2>&1 | tail -n 4 | tee -a "$LOG"
echo "===================== TABLE1 (4列) DONE =====================" | tee -a "$LOG"
