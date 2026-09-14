#!/usr/bin/env bash
# watch_ood_abc.sh — 48h grid watchdog (survives SSH disconnect; run with nohup).
#
# Every 5 min, for each of the 6 arms (cifar10/gtsrb x a/b/c, seed 1):
#   - detect state: QUEUED / RUNNING(epoch n) / DONE(epoch 299) / FAILED(launcher exit!=0)
#   - on DONE: extract final + last-20-mean PoisonACC(=ASR) & CleanACC(=BA),
#     copy the log + trigger meta to the cross-disk backup dir (once)
#   - rewrite results_ood/STATUS.md (the autonomous "report": user reads this on return)
# Exit when all 6 arms are DONE/FAILED, or after 60h hard timeout.

BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
# Full tag list: 6 primary arms (seed1) + 2 cur references + 4 seed2 preload runs
ARMS="cifar10_a_seed1 cifar10_b_seed1 cifar10_c_seed1 cifar10_cur_seed1 cifar10_a_seed2 cifar10_c_seed2 \
      gtsrb_a_seed1 gtsrb_b_seed1 gtsrb_c_seed1 gtsrb_cur_seed1 gtsrb_a_seed2 gtsrb_c_seed2 \
      gtsrb_a_seed1_ep150 gtsrb_b_seed1_ep150 gtsrb_c_seed1_ep150 \
      gtsrb_a_seed1_pr0.0005 gtsrb_a_seed2_pr0.0005 gtsrb_a_seed1_pr0.001 gtsrb_a_seed2_pr0.001 \
      gtsrb_a_seed1_pr0.0025 gtsrb_a_seed2_pr0.0025 gtsrb_a_seed1_pr0.005 gtsrb_a_seed2_pr0.005 \
      cifar10_a_seed1_ep150 cifar10_b_seed1_ep150 cifar10_c_seed1_ep150 \
      gtsrb_b_seed2 gtsrb_cur_seed2 \
      gtsrb_c_seed1_w0.5 gtsrb_c_seed2_w0.5 gtsrb_c_seed1_w0.25 gtsrb_c_seed2_w0.25 \
      gtsrb_cur_seed1_pr0.0005 gtsrb_cur_seed2_pr0.0005 gtsrb_cur_seed1_pr0.001 gtsrb_cur_seed2_pr0.001 \
      gtsrb_cur_seed1_pr0.005 gtsrb_cur_seed2_pr0.005 \
      gtsrb_c_seed1_w0.05 gtsrb_c_seed2_w0.05 gtsrb_c_seed3_w0.05 \
      gtsrb_c_seed1_w0.1 gtsrb_c_seed2_w0.1 gtsrb_c_seed3_w0.1 \
      gtsrb_c_seed1_w0.15 gtsrb_c_seed2_w0.15 gtsrb_c_seed3_w0.15 \
      cifar100_a_seed1 cifar100_b_seed1 cifar100_c_seed1 cifar100_cur_seed1"
BK=/media/hd0/wangxin/backup/ood_abc_20260911
STATUS=results_ood/STATUS.md
DEADLINE=$(( $(date +%s) + 60*3600 ))

mkdir -p "$BK" results_ood

epoch_metrics() {  # $1=log  ->  "epoch asr_f ba_f asr20 ba20" or ""
  local log="$1"
  [ -f "$log" ] || { echo ""; return; }
  grep -E '\] - [0-9]+[[:space:]]' "$log" | tail -20 | awk -F'\t' '
    { sub(/.*- /,"",$1); ep=$1+0; s7+=$7; s9+=$9; n++
      if (ep>maxep) { maxep=ep; f7=$7; f9=$9 } }
    END { if (n>0) printf "%d %.4f %.4f %.4f %.4f", maxep, f7, f9, s7/n, s9/n }'
}

target_ep() {  # $1=arm tag -> final epoch index (149 for _ep150 screening runs, else 299)
  case "$1" in *_ep150) echo 149 ;; *) echo 299 ;; esac
}

all_settled() {
  local a log te
  for a in $ARMS; do
    log=$(ls -t results_ood/oodabc_${a}/output_*.log 2>/dev/null | head -1)
    [ -n "$log" ] || return 1
    te=$(target_ep "$a")
    grep -qsE "\] - ${te}[[:space:]]" "$log" || return 1
  done
  return 0
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  {
    echo "# OOD-ABC 48h 网格 — 自动状态报告（watchdog 每 5 分钟更新）"
    echo
    echo "更新时间: $(date '+%F %T')　|　分支: ood-calibration@main　|　参照锚点: FAAT v4 — CIFAR-10 93.6 / GTSRB(l2=3.5) 82.7"
    echo
    echo "| run | 状态 | epoch | ASR(末值) | ASR(末20均) | BA(末值) | BA(末20均) |"
    echo "|---|---|---|---|---|---|---|"
    for a in $ARMS; do
      tag="oodabc_${a}"
      log=$(ls -t results_ood/${tag}/output_*.log 2>/dev/null | head -1)
      m=$(epoch_metrics "$log")
      if [ -z "$m" ]; then
        st="QUEUED（等前序臂）"
        row="| $tag | $st | - | - | - | - | - |"
      else
        set -- $m
        ep=$1; asr_f=$2; ba_f=$3; asr20=$4; ba20=$5
        te=$(target_ep "$a")
        if grep -qsE "\] - ${te}[[:space:]]" "$log"; then
          if [ -f "$BK/${tag}_$(basename "$log")" ]; then bk="DONE ✅已备份"
          else
            cp "$log" "$BK/${tag}_$(basename "$log")" 2>/dev/null
            [ -f "resource_ood/triggers/${tag}/global_meta.json" ] && \
              cp "resource_ood/triggers/${tag}/global_meta.json" "$BK/${tag}_global_meta.json" 2>/dev/null
            bk="DONE ✅已备份"
          fi
          row="| $tag | $bk | $ep | $asr_f | $asr20 | $ba_f | $ba20 |"
        else
          # FAILED 检测：启动器日志该臂已 finished 且 exit!=0
          if grep -qs "finished $tag (exit [1-9]" results_ood/_abc_*.log; then
            row="| $tag | ❌ FAILED（启动器非零退出，见 _abc_*.log） | $ep | $asr_f | $asr20 | $ba_f | $ba20 |"
          else
            row="| $tag | RUNNING | $ep/$((te+1)) | $asr_f | $asr20 | $ba_f | $ba20 |"
          fi
        fi
      fi
      echo "$row"
    done
    echo
    echo "**Go/No-Go 速查**（同 L2 budget 下比 ASR，BA 不得明显跌；GTSRB 是生死场）："
    echo "- c vs a：校准项带来增益？　- c vs b：增益来自\"校准\"而非\"多了数据\"？（c≈b = NO-GO）"
    echo "- 全部完成且某臂存活 → 补 seed2/3；只在 CIFAR-10 有效 → NO-GO。"
    echo
    echo "日志: \`results_ood/oodabc_*/output_1.log\`　启动器: \`results_ood/_abc_*_gpu*.log\`　备份: \`$BK\`"
  } > "$STATUS"

  if all_settled; then
    echo "[watchdog $(date '+%F %T')] all arms settled -> final STATUS.md written, exiting" >> results_ood/_watchdog.log
    cp results_ood/_abc_*.log "$BK/" 2>/dev/null
    exit 0
  fi
  sleep 300
done
echo "[watchdog $(date '+%F %T')] 60h timeout reached, exiting" >> results_ood/_watchdog.log
