#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 train_backdoor.py 的 output_1.log，提取每个选择策略的 ASR 与 BA。
日志列: Epoch lr Time TrainLoss TrainACC PoisonLoss PoisonACC CleanLoss CleanACC TargetSum CleanSum
  PoisonACC = ASR (触发后非目标类测试图被预测为目标的比率)
  CleanACC  = BA  (干净测试精度)
报告: 最后一个 epoch 的 ASR/BA, 以及最后 20 个 epoch 的均值(更稳)。
"""
import os, re, sys, glob

def parse_log(path):
    """返回 (last_epoch, last_asr, last_ba, mean20_asr, mean20_ba, n_epochs) 或 None"""
    if not os.path.exists(path):
        return None
    epoch_pat = re.compile(r"^\s*\[(.+?)\] - (\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    rows = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = epoch_pat.match(line)
            if m:
                g = m.groups()
                ep = int(g[1])
                # 列: 1=epoch 2=lr 3=time 4=trainloss 5=trainacc 6=poisonloss 7=poisonacc 8=cleanloss 9=cleanacc 10=targetsum 11=cleansum
                poison_acc = float(g[7])   # PoisonACC = ASR
                clean_acc  = float(g[9])   # CleanACC  = BA
                rows.append((ep, poison_acc, clean_acc))
    if not rows:
        return None
    last = rows[-1]
    tail = rows[-20:]
    mean_asr = sum(r[1] for r in tail) / len(tail)
    mean_ba  = sum(r[2] for r in tail) / len(tail)
    return (last[0], last[1]*100, last[2]*100, mean_asr*100, mean_ba*100, len(rows))

# 论文 Table 1 MultiBpp-RGB (CIFAR-10, 1%) 目标值
PAPER = {
    "quantize_random":   ("Random",   1.16, 94.95),
    "quantize_loss":     ("Loss",    47.85, 94.76),
    "quantize_gradient": ("Gradient",53.28, 95.03),
    "quantize_forget":   ("Forget",  78.10, 94.90),
    "quantize_res_log":    ("Res-log",  80.20, 94.82),
    "quantize_res_linear": ("Res-x",    83.07, 94.63),
    "quantize_res_square": ("Res-x^2",  83.88, 94.59),
    "quantize_res_exp":    ("Res-e^x",  62.28, 94.85),
}

if __name__ == "__main__":
    resdir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    print(f"{'dir':<24}{'sel':<10}{'epoch':>7}{'ASR_last':>10}{'ASR_m20':>10}{'BA_last':>10}{'BA_m20':>10}   | {'paper_ASR':>9} {'paper_BA':>8} {'gap_ASR':>8}")
    print("-"*120)
    for d in sorted(glob.glob(os.path.join(resdir, "quantize_*"))):
        name = os.path.basename(d)
        r = parse_log(os.path.join(d, "output_1.log"))
        if r is None:
            print(f"{name:<24} (无日志/未开始)")
            continue
        ep, lasr, lba, masr, mba, n = r
        paper = PAPER.get(name, ("?", None, None))
        gap = f"{lasr-paper[1]:+.2f}" if paper[1] is not None else "?"
        pp = f"{paper[1]:.2f}" if paper[1] is not None else "?"
        pb = f"{paper[2]:.2f}" if paper[2] is not None else "?"
        print(f"{name:<24}{paper[0]:<10}{ep:>7}{lasr:>10.2f}{masr:>10.2f}{lba:>10.2f}{mba:>10.2f}   | {pp:>9} {pb:>8} {gap:>8}")
