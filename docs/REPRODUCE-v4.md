# FAAT v4 精确复现命令（自包含 Narcissus —— 去掉对作者 noise_01000.pth 的依赖）

> **v4 的核心改变**：v3.1 的 δ_global 来自作者的 `noise_01000.pth`（CIFAR-10 专用，3×32×32/10 类），
> 因此无法迁移——GTSRB 崩到 ASR 0.9%。v4 用**自包含 Narcissus 引擎**从零优化 universal δ_global
> （CE 骗干净代理 + L2 投影 + 多步），对**任意数据集**都成立，完全不碰作者的 noise 文件。
> 下游全部复用 v3.1：bounded adaptive（≤0.15）+ L_align + 注入 + 训练 + 防御。
>
> 代码快照：`snapshots/v4_2026-07-04/`；git 提交：`351313a`（分支 `exp/v4-self-contained`）。
> 关键 flag：`--global_mode from_scratch`（启用引擎）+ `--fix_global`（冻结生成结果）+ `--global_l2_budget`（触发器 L2 预算=隐蔽预算）。

## proxy-ASR sanity（无 victim 训练，仅验引擎能否骗过干净代理）
```bash
# CIFAR-10：L2=1.5 -> proxy-ASR ≈ 0.89（仅用作者 noise 的 gs010 scale0.1 只有 0.71）
python -m faat.global_trigger --dataset cifar10 --data_dir ./data --num_classes 10 --size 32 \
  --y_target 0 --seed 1 --proxy_path resource/faat/proxy/resnet18_clean_cifar10.pth \
  --l2_budget 1.5 --steps 8000 --lr 0.02 --loss ce --init zero \
  --save_dir resource/faat/v4/cifar10/sanity_l2_1.5

# GTSRB（43 类）：L2 与 proxy-ASR 单调 —— 1.5->0.43 / 2.5->0.70 / 3.5->0.85 / 5.0->0.96
python -m faat.global_trigger --dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 --size 32 \
  --y_target 0 --seed 1 --proxy_path resource/faat/proxy/resnet18_clean_gtsrb.pth \
  --l2_budget 3.5 --steps 10000 --lr 0.02 --loss ce --init zero \
  --save_dir resource/faat/v4/gtsrb/sanity_l2_3.5
```

## 全量 FAAT（自包含 δ_global + 有界 adaptive，CIFAR-10 冠军配置）
```bash
CUDA_VISIBLE_DEVICES=0 python -u train_faat.py \
  --dataset cifar10 --data_dir ./data --num_classes 10 --size 32 --y_target 0 \
  --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_10_res --select_epoch 10 --seed 1 --device cuda \
  --proxy_path ./resource/faat/proxy/resnet18_clean_cifar10.pth --train_proxy_if_missing \
  --save_trigger ./resource/faat/v4/cifar10/l2_1.5_seed1 \
  --result_dir results/faatb_v4_cifar10_l2_1.5_seed1 \
  --global_mode from_scratch --global_l2_budget 1.5 --global_steps 8000 --global_lr 0.02 --global_loss ce \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --steps 2000 --batch_size 48 --train --epochs 300
```

## GTSRB 跨数据集（43 类，预算需更大）
```bash
CUDA_VISIBLE_DEVICES=2 python -u train_faat.py \
  --dataset gtsrb --data_dir data/GTSRB32 --num_classes 43 --size 32 --y_target 0 \
  --selection res --res_sel square --poison_rate 0.01 \
  --output_dir ./resource/save_metric_gtsrb --select_epoch 10 --seed 1 --device cuda \
  --proxy_path ./resource/faat/proxy/resnet18_clean_gtsrb.pth --train_proxy_if_missing \
  --save_trigger ./resource/faat/v4/gtsrb/l2_3.5_seed1 \
  --result_dir results/faatb_v4_gtsrb_l2_3.5_seed1 \
  --global_mode from_scratch --global_l2_budget 3.5 --global_steps 10000 --global_lr 0.02 --global_loss ce \
  --fix_global --adaptive_l2_max 0.15 --eps_max 0.05 \
  --lambda_align 1.0 --lambda_perc 0.3 --lambda_l2 0.05 --lambda_freq 0.02 \
  --steps 2000 --batch_size 48 --train --epochs 300
```

## GPU 感知批量调度（队列 + 自动派卡 + 每实验 banner）
```bash
# 1) 写队列 JSON（每条 spec = 一个实验，见 logs/v4/queue.json 模板）
# 2) 启动调度器：监听空闲卡（free≥8GB & util≤30%），排除 busy/指定卡，每 run 打印"当前实验组"
python -m faat.scheduler --queue logs/v4/queue.json --exclude 1 --poll 20
# 干跑（只打印命令不执行）：加 --dry_run
```

## v4 相对 v3.1 的 flag 差异
| flag | v3.1 | v4 |
|---|---|---|
| `--global_mode` | （无，恒用 nar_file） | `from_scratch` |
| `--global_l2_budget` | （无） | 触发器 L2 预算（CIFAR 1.5 / GTSRB 3.5+） |
| `--init_global_scale` | 控制 Narcissus 缩放 | 失效（from_scratch 下预算已烘焙，恒 1.0） |
| δ_global 来源 | `noise_01000.pth` | 自包含引擎从零优化 |
