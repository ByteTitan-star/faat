#!/bin/bash
# run_tiny_chain.sh — 自主串联 Tiny-ImageNet 实验 (cal_metric 等待 → proxy → FAAT v7 + baselines v7b),
# GPU1 全程排除。
#
#   cal_metric (3 seed, 已在跑)  →  train clean proxy (GPU0)  →  merged scheduler:
#     v7  (FAAT 0.25%,  9 组)  ← 方案A 直比论文 Table7 (43.93) + matched
#     v7b (baselines 0.25%, 9 组) ← 方案B matched 对比
#   合并成单队列跑单 scheduler：3 GPU 全程满载、无 race、FAAT 优先(headline 先出)。
#   每步写里程碑到 logs/tiny_chain.log。Recording(eval_all) 在各里程碑由主控做。
set -u
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
CHAIN_LOG=logs/tiny_chain.log
METRIC_DIR=resource/save_metric_tiny_res
mkdir -p logs

# --- cal_metric 完成判定: 3 个 seed 都有 stats_forget + epoch_10 pkl ---
cal_done() {
  n=0
  for sd in 1 2 3; do
    if [ -f "$METRIC_DIR/stats_forget_seed_$sd.pkl" ] && \
       [ -f "$METRIC_DIR/resnet_loss_grad_epoch_10_seed_$sd.pkl" ]; then
      n=$((n+1))
    fi
  done
  echo $n
}

echo "[$(date '+%F %T')] === Tiny-ImageNet chain 启动 (GPU1 全程排除) ===" | tee -a $CHAIN_LOG

# --- 1. 等 cal_metric (3 seed) 跑完 ---
echo "[$(date '+%F %T')] 等待 cal_metric 完成 (3/3 seed: stats_forget + epoch_10 pkl) ..." | tee -a $CHAIN_LOG
while [ "$(cal_done)" != "3" ]; do sleep 60; done
echo "[$(date '+%F %T')] cal_metric DONE (3/3). 训练 clean proxy (GPU0, 100ep) ..." | tee -a $CHAIN_LOG

# --- 2. 训练 clean proxy (foreground, GPU0) ---
PROXY=resource/faat/proxy/resnet18_clean_tiny.pth
CUDA_VISIBLE_DEVICES=0 $PY -c "
from faat.proxy import train_clean_proxy
train_clean_proxy(dataset='tiny', num_classes=200, data_dir='./data_tiny',
                  epochs=100, lr=0.1, batch_size=128, device='cuda',
                  save_path='$PROXY', seed=1)
" >> $CHAIN_LOG 2>&1
if [ ! -f "$PROXY" ]; then
  echo "[$(date '+%F %T')] !! proxy 训练失败, 中止 chain" | tee -a $CHAIN_LOG
  exit 1
fi
echo "[$(date '+%F %T')] proxy DONE -> $PROXY. 合并 v7(FAAT)+v7b(baseline) 队列 ..." | tee -a $CHAIN_LOG

# --- 3. 合并 v7 (FAAT, 优先) + v7b (baseline) 成单队列 ---
$PY -c "
import json
v7  = json.load(open('logs/v7_tiny_faat/queue.json'))
v7b = json.load(open('logs/v7b_tiny_baselines/queue.json'))
merged = v7 + v7b   # FAAT first (headline), then matched baselines
json.dump(merged, open('logs/v7_tiny_merged/queue.json','w'), indent=2)
print('merged %d FAAT + %d baseline = %d runs' % (len(v7), len(v7b), len(merged)))
" >> $CHAIN_LOG 2>&1
mkdir -p logs/v7_tiny_merged

# --- 4. 单 scheduler 跑合并队列 (exclude GPU1) ---
$PY -m faat.scheduler --queue logs/v7_tiny_merged/queue.json \
   --log_dir logs/v7_tiny_merged --state logs/v7_tiny_merged/scheduler_state.json \
   --exclude 1 --poll 20 >> $CHAIN_LOG 2>&1
echo "[$(date '+%F %T')] scheduler DONE. 开始记录实验数据 (eval_all FAAT + baseline parse) ..." | tee -a $CHAIN_LOG

# --- 5. 记录 FAAT 结果 (eval_all: ASR/BA + 隐蔽 SSIM/L2/DCT + 防御 AC/SS/STRIP/FP) ---
#    eval_all 幂等: 只对缺 stageB/defenses.json 的 run 跑重指标; ASR/BA 永远从日志取。
CUDA_VISIBLE_DEVICES=0 $PY -m faat.eval_all \
   --pattern 'faatb_v7_*' --trigger_version v7 --log_dir logs/v7_tiny_merged \
   --out_csv docs/v7_results.csv --out_json docs/v7_results.json >> $CHAIN_LOG 2>&1
echo "[$(date '+%F %T')] FAAT 记录 DONE -> docs/v7_results.csv/.json" | tee -a $CHAIN_LOG

# --- 6. 记录 baseline 结果 (ASR/BA 从训练日志末20epoch均, 同 v6b 口径) ---
$PY -c "
import re, csv, json, glob, os
LOG='logs/v7_tiny_merged'
def last20(path):
    # strip '[date] - ' prefix, then cols: Epoch lr Time TrainLoss TrainACC PoisonLoss PoisonACC(ASR) CleanLoss CleanACC(BA)
    rows=[]
    for ln in open(path):
        m=re.search(r'\] - (.+)$', ln); body=m.group(1) if m else ln
        t=body.split()
        if len(t)>=9 and re.match(r'^\d+$', t[0]):
            rows.append((float(t[6]), float(t[8])))
    if not rows: return None,None
    last=rows[-20:]
    return round(sum(x[0] for x in last)/len(last)*100,1), round(sum(x[1] for x in last)/len(last)*100,2)
recs=[]
for rdir in sorted(glob.glob('results/bl_tiny_*_seed*')):
    name=os.path.basename(rdir)
    logp=os.path.join(LOG, name+'.log')
    asr,ba=last20(logp) if os.path.exists(logp) else (None,None)
    recs.append({'run':name,'ASR':asr,'BA':ba})
    print('  %-28s ASR=%s BA=%s'%(name,asr,ba))
with open('docs/v7b_baselines_results.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['run','ASR','BA']); w.writeheader(); w.writerows(recs)
json.dump(recs,open('docs/v7b_baselines_results.json','w'),indent=2)
print('wrote %d baseline records -> docs/v7b_baselines_results.csv'%len(recs))
" >> $CHAIN_LOG 2>&1
echo "[$(date '+%F %T')] baseline 记录 DONE -> docs/v7b_baselines_results.csv/.json" | tee -a $CHAIN_LOG

echo "[$(date '+%F %T')] === chain ALL DONE + RECORDED: cal_metric + proxy + v7(9 FAAT) + v7b(9 baseline) ===" | tee -a $CHAIN_LOG
