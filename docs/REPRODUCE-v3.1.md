# FAAT v3.1 精确复现命令（代码即使后续被 v4 改动，凭此 + git 快照也能重跑 v3.1）

> 代码快照：`snapshots/v3.1_2026-07-03/`；git 提交：`a2aab8f`（v3.1 baseline）。
> v3.1 正解 = 固定 Narcissus δ_global（`--fix_global`，保留作者强 ASR 方向）+ 有界学习型 δ_adaptive（`--adaptive_l2_max 0.15`，提升低预算 ASR 效率、不夺权破坏 C2）。

## 冠军配置（CIFAR-10, scale 0.2，三目标全达标）
```bash
CUDA_VISIBLE_DEVICES=0 python -u train_faat.py \
  --dataset cifar10 --y_target 0 --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --proxy_path ./resource/faat/proxy/resnet18_clean_cifar10.pth \
  --steps 2000 --batch_size 48 \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --init_global_scale 0.2 \
  --save_trigger ./resource/faat/v3_1/scale_0.2 \
  --train --gpu 0 --result_dir results/faatb_v3_1_scale_0.2
```
结果(末20均)：ASR 94.81 / BA 94.67 / L2 1.31 / SSIM 0.963 / AC-AUC 0.22 / SS-AUC 0.42。

## scale sweep（换 `--init_global_scale`）
0.17(更隐蔽冠军): ASR 90.45 / L2 1.12 / SSIM 0.972｜0.5: 99.82｜0.1: 70.96(>Narcissus gs010 68.57)

## 关键 flag 含义（v3.1 = 下列组合；去掉对应 flag 可退化到旧版本）
- `--fix_global`：δ_global 冻结=Narcissus×scale，不优化（v3.1 核心）。
- `--adaptive_l2_max 0.15`：δ_adaptive 硬 L2 上界（防 v3 的无界 adaptive 喧宾夺主）。
- 去掉 `--fix_global` 改加 `--global_obj align` → v1；`--global_obj asr` → v2。
- 加 `--global_l2_max X`：对 δ_global 施加硬 L2 预算（Pareto sweep 用）。

## 复现历史版本（代码是累加 flag，旧版本仍可复现）
- v1(L_align 优化 δ_global)：`--global_obj align`（无 fix_global）
- v2(CE 骗代理)：`--global_obj asr`（无 fix_global）
- v3(无界 adaptive，已证失败)：`--fix_global`（不加 adaptive_l2_max）
- v3.1(当前最佳)：`--fix_global --adaptive_l2_max 0.15`

## 隐蔽/检测度量
```bash
python -m faat.stage_b_metrics --device cuda \
  --save_trigger resource/faat/v3_1/scale_0.2 --rdir results/faatb_v3_1_scale_0.2
```

## GTSRB 跨数据集（无现成 Narcissus → 自生成触发器）
```bash
python -m faat.gtsrb_prep   # 一次性：Resize32 + 90/10 切分 → data/GTSRB32
python cal_metric.py --dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 --seed 1 --epochs 11 --output_dir ./resource/save_metric_gtsrb
CUDA_VISIBLE_DEVICES=3 python -u train_faat.py \
  --dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 --y_target 0 --selection res --res_sel square \
  --poison_rate 0.01 --output_dir ./resource/save_metric_gtsrb --select_epoch 10 --seed 1 --device cuda \
  --train_proxy_if_missing --proxy_path ./resource/faat/proxy/resnet18_clean_gtsrb.pth \
  --steps 2000 --batch_size 48 --global_obj asr --init_random --global_l2_max 1.3 --adaptive_l2_max 0.15 \
  --eps_max 0.05 --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --save_trigger ./resource/faat/gtsrb/l2_1.3 --train --gpu 3 --result_dir results/faatb_gtsrb_l2_1.3
```
