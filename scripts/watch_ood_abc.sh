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
SEED=1
ARMS="cifar10_a cifar10_b cifar10_c gtsrb_a gtsrb_b gtsrb_c"
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

all_settled() {
  local a log
  for a in $ARMS; do
    log="results_ood/oodabc_${a}_seed${SEED}/output_1.log"
    grep -qsE '\] - 299[[:space:]]' "$log" || return 1
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
      tag="oodabc_${a}_seed${SEED}"
      log="results_ood/${tag}/output_1.log"
      m=$(epoch_metrics "$log")
      if [ -z "$m" ]; then
        st="QUEUED（等前序臂）"
        row="| $tag | $st | - | - | - | - | - |"
      else
        set -- $m
        ep=$1; asr_f=$2; ba_f=$3; asr20=$4; ba20=$5
        if grep -qsE '\] - 299[[:space:]]' "$log"; then
          if [ -f "$BK/${tag}_output_1.log" ]; then bk="DONE ✅已备份"
          else
            cp "$log" "$BK/${tag}_output_1.log" 2>/dev/null
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
            row="| $tag | RUNNING | $ep/300 | $asr_f | $asr20 | $ba_f | $ba20 |"
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
    cp results_ood/_abc_cifar10_gpu2.log results_ood/_abc_gtsrb_gpu1.log "$BK/" 2>/dev/null
    exit 0
  fi
  sleep 300
done
echo "[watchdog $(date '+%F %T')] 60h timeout reached, exiting" >> results_ood/_watchdog.log
