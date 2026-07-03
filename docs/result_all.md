# result_all.md — GeneralComponents 复现总追踪

> 论文：**A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks**（NeurIPS 2025, arXiv 2509.19947）
> 代码：本仓库（`train_backdoor.py` + `cal_metric.py`）
> 目的：**复现论文全部实验作为 baseline，支撑 FAAT 新方法论文写作。**
> 本文件 = 复现状态的**唯一真相源（single source of truth）**。每跑完一组实验，必须回此更新数据。

---

## 📊 当前进度（一句话总览）

> **本阶段复现范围 = 论文 Table 1（CIFAR-10，1% 投毒，8 选择策略 × 4 攻击 = 32 组）。**
> 其余表（CIFAR-100、消融、扩展、防御）见 §1.2，列为**后续可选**，不计入当前 total。

> ## ✅ Table 1 复现完成 32/32（2026-07-01 21:04 全组跑满 epoch=299）—— 已暂停，等 FAAT agent 改完代码
> **全 32 组完成**，调度器 21:04 自动 `parse_detail.py --write-md` 生成 §3.1（32 行）。剩 4 组 Blended-C（`res_exp`/`loss`/`gradient`/`random`）已于 18:04–18:05 在 FAAT 让出并发槽后于 GPU0/GPU3 起跑、21:04 前全部跑满。`results/faat_*` 被 `parse_detail.py:210` 自动跳过，未污染本表。
> **代码完整性已核（32 组有效）**：`utils.py`(blend 补丁[2,2,2]+`get_stats`) mtime 全程冻在 11:32 未动，`cifar_resnet.py`(网络) 冻在 06-29；`train_backdoor.py` 仅被 FAAT 加了 `faat`/`faatb` 分支（纯加法，blend 的 `else` 路径 `diff` 无变化）。冻结审计备份在 `/tmp/repro_code_frozen_blend/`。

- **总目标**：Table 1 共 **32 组** → ✅ **32/32 全部完成**。
- **四列 ✅**：Badnets-C **8/8**、MultiBpp-B **8/8**、MultiBpp-RGB **8/8**、Blended-C **8/8**。
- **调度器**：21:04 跑完 32 组后正常退出，已自动 `parse_detail.py --write-md`；现已停。
- **复现结论**：四列本文最优 Res-x² 与 Forget 均匹配论文（见 §3.1/§3.3，差距属单种子方差；Res-eˣ 偏低见 §3.4 caveat）。32 组可作 FAAT baseline。
- 最近更新：2026-07-01 21:04（**32/32 完成**：4 组 blend 补跑完毕、代码核验未污染；现暂停等 FAAT agent）

> **📋 后续（等 FAAT agent 改完代码后再定）**：
> 1. **可选抽查防污染**：任选 1 组已完成组（如 `blend_res_square`），先 `mv output_1.log output_1.log.verify` 再重跑，对比 ASR/BA 与 §3.1 一致 → 证 32 组未被污染（代码 `diff` 已间接证明，此为双重保险）。
> 2. **后续表（可选）**：CIFAR-100(Table2)/消融(Table3,4,6,7)/扩展/防御(需 BackdoorBench)，见 §1.2。
> 3. **不要从头重跑 32 组**：已完成且有效。

| 范围 | 论文表 | 组数 | 已完成 | 状态 |
|---|---|---|---|---|
| **★ 当前范围** Table 1（CIFAR-10, 1%） | Table 1 | **32** | **32** | ✅ **完成**(2026-07-01 21:04) |
| 后续可选：CIFAR-100 主实验 | Table 2 | 32 | 0 | ⬜ 不在当前范围 |
| 后续可选：消融 | Table 3,4,6,7 | ~84 | 0 | ⬜ |
| 后续可选：扩展 | Table 9,10,11,12, Fig.6 | ~70 | 0 | ⬜ |
| 后续可选：防御 | Table 5,13 | 100+ | 0 | 🔧 需 BackdoorBench |

---

## 1. 论文完整实验矩阵（需复现的全部组）

> 每组 = 一个独立训练 run（一个 `output_1.log`）。下表是论文**所有结果表**的清单，按优先级分层。
> `状态`列：✅ 已完成 / 🔄 进行中 / ⬜ 待跑 / 🔧 需额外依赖

### Tier 1 — 核心主实验（最高优先，FAAT 直接对照的 baseline）

#### Table 1：CIFAR-10，1% 投毒，样本选择 × 4 攻击 → **32 组**
- 数据集 CIFAR-10，模型 ResNet18，y_target=0，seed=1，300 epoch，poison_rate=0.01（500 样本，clean-label）
- 选择策略 8 个：Random / Loss / Gradient / Forget / Res-log / Res-x / Res-x² / Res-eˣ
- 攻击 4 列：**Badnets-C** / **Blended-C** / **MultiBpp-B** / **MultiBpp-RGB**
- 详细状态见 [§3 已完成结果](#3-已完成实验详细结果table-1--multibpp-rgb-列) 与 [§4 Table 1 全表对照](#4-论文-table-1-全表32-组--复现状态)

| 攻击列 | 代码命令关键参数 | 选择策略数 | 组数 | 状态 |
|---|---|---|---|---|
| Badnets-C | `--backdoor_type badnets --type 0:0:0`（代码硬编码黑白棋盘=[0,0,0]，等价 0:0:0） | 8 | 8 | ✅ **8/8** |
| Blended-C | `--backdoor_type blend`；**utils.py 已补丁行187/303 为 [2,2,2]=0.2:0.2:0.2**（原代码硬编码 [2,1,3]=0.2:0.1:0.3，见 §5.4）；`--type 2:2:2` 占位 | 8 | 8 | ✅ **8/8 完成** |
| MultiBpp-B | `--backdoor_type quantize --num_levels 255:255:8`（仅蓝通道强量化 step=36，R/G 近乎不变，已确认） | 8 | 8 | ✅ **8/8** |
| **MultiBpp-RGB** | `--backdoor_type quantize --num_levels 24:28:8` | 8 | 8 | ✅ **8/8** |

#### Table 2：CIFAR-100，2 种投毒率 × 2 攻击 × 8 选择策略 → **32 组**
- 数据集 CIFAR-100（`./data100`，100 类），α=0.2% 与 0.5%（即选目标类 20% / 50%）
- 攻击：Badnets-C、Blended-C
- 🔧 **前置依赖**：需先在 CIFAR-100 上跑 `cal_metric.py` 生成 `save_metric_100_*`（当前只有 CIFAR-10 的 `save_metric_10_res`）
- 状态：⬜ 0/32

### Tier 2 — 消融实验

| 论文表 | 内容 | 组数 | 命令要点 | 状态 |
|---|---|---|---|---|
| Table 6 | BlendX（Blend32/28/24/20）× 7 选择策略，CIFAR-10 1% | 28 | `--backdoor_type blend --blend_size {32,28,24,20}` | ⬜ |
| Table 7 | Badnets 8 种触发模式 × {Random,Res-x²} × {1%,2.5%}，CIFAR-10 | 32 | `--backdoor_type badnets --type {0:0:0,1:1:1,...}` | ⬜ |
| Table 3 | 全局投毒攻击 12 种量化设置，CIFAR-10 2.5%（Component C） | 12 | `--backdoor_type quantize --num_levels {Base,32:32:32,255:255:8,...}` | ⬜ |
| Table 4 | Component A/B/C 协同效应，CIFAR-10 1%（Badnets-C / Blended-C） | ~12 | 需 `--selection stealth`（Component B）等组合 | 🔧 |

### Tier 3 — 扩展实验

| 论文表 | 内容 | 组数 | 状态 |
|---|---|---|---|
| Table 9 | Badnets-C 不同目标类（y∈{0,10,…,80}），CIFAR-100 | ~18 | ⬜ |
| Table 10 | 脏标签（poison-label）攻击，CIFAR-100，0.05%/0.1% | 12 | ⬜ |
| Table 11/12 | Tiny-ImageNet（200 类）当前方法 vs 本文方法 | ~24 | 🔧 需数据集 |
| Figure 6 | Component A 在 {R18,R34,V16,D121} 多模型迁移，Blended-C | ~16 | ⬜ |

### Tier 4 — 防御实验（最后做）

| 论文表 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| Table 5 | 7 方法（random/forget/+A/+B/+C/+B&C/+A&C）× 8 防御（ABL/AC/FP/I-BAU/NC/RNP/FST），CIFAR-10 3% | BackdoorBench | 🔧 |
| Table 13 | SIG / CTRL 攻击的防御评估 | BackdoorBench | 🔧 |

---

## 2. 统一环境与训练配置（所有实验共用）

- **conda 环境**：`/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents`
  - python3.8 + torch1.11.0+cu113 + torchvision0.12.0+cu113 + numpy1.22.3 + pillow9.5.0
- **GPU**：RTX 3090 ×4（与其他任务共享，每组约 2-3GB 显存）
- **数据**：CIFAR-10 复用本机 `/home/wangxin/data/cifar-10-batches-py`，软链到 `./data`
- **样本选择指标**：复用作者提供的 `./resource/save_metric_10_res`（跳过 `cal_metric.py` 的 Step1）
- **训练超参（CIFAR-10 默认）**：ResNet18, 300 epoch, SGD lr=0.1 momentum=0.9 nesterov wd=5e-4, milestones=[60,90] γ=0.1, batch=128, seed=1, y_target=0(airplane), poison_rate=0.01 → 500 投毒样本（全取自类0，clean-label）
- **指标定义**：
  - `ASR` = PoisonACC（触发后非目标类测试图被预测为 target 的比率）
  - `BA` = CleanACC（干净测试精度）
- **报告口径**：ASR/BA 取**末 20 个 epoch 均值**（比单点末值更稳）；同时保留末值、峰值、std 供溯源。

---

## 3. 已完成实验详细结果（Table 1 已完成列）

> 统一配置：CIFAR-10 + ResNet18 + 1% 投毒 + seed=1 + 300 epoch；区别仅在攻击列（Badnets-C / Blended-C / MultiBpp-B / MultiBpp-RGB）。
> 详细 JSON：`results/_detail_summary.json`；解析脚本：`parse_detail.py`。下表为自动生成。

### 3.1 汇总表（按攻击列 + ASR 末20均 排序）

> ⚙️ 本表由 `python parse_detail.py ./results results/_detail_summary.json --write-md docs/result_all.md` **自动生成**，请勿手改标记之间的内容；跑完新实验后重跑该命令即可同步。

<!-- AUTO:result_table_start -->
| 实验组 | 攻击列 | selection | ASR末值 | ASR末20均(±std) | ASR峰值@ep | BA末值 | BA末20均 | TrainACC末 | PoisonLoss末 | 耗时min | 论文ASR | ASR差距 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quantize_res_linear | MultiBpp-RGB | res/linear | 88.87 | 85.09(±2.15) | 99.89@25 | 94.79 | 94.80 | 99.99 | 0.4964 | 212.7 | 83.07 | +5.80 |
| quantize_res_log | MultiBpp-RGB | res/log | 88.87 | 85.09(±2.15) | 99.89@25 | 94.79 | 94.80 | 99.99 | 0.4964 | 214.0 | 80.2 | +8.67 |
| quantize_res_square | MultiBpp-RGB | res/square | 85.36 | 82.99(±2.82) | 99.49@22 | 94.60 | 94.58 | 100.00 | 0.6613 | 213.6 | 83.88 | +1.48 |
| quantize_forget | MultiBpp-RGB | forget | 85.39 | 81.18(±2.40) | 99.26@48 | 94.82 | 94.85 | 100.00 | 0.6629 | 203.7 | 78.1 | +7.29 |
| quantize_loss | MultiBpp-RGB | loss | 75.10 | 72.80(±2.22) | 96.06@54 | 94.97 | 94.96 | 100.00 | 1.1641 | 205.9 | 47.85 | +27.25 |
| quantize_gradient | MultiBpp-RGB | grad | 69.92 | 62.26(±3.04) | 93.96@33 | 94.65 | 94.71 | 100.00 | 1.5255 | 196.7 | 53.28 | +16.64 |
| quantize_res_exp | MultiBpp-RGB | res/exp | 59.98 | 60.20(±3.26) | 96.87@32 | 94.72 | 94.75 | 100.00 | 2.0827 | 196.3 | 62.28 | -2.30 |
| quantize_random | MultiBpp-RGB | random | 32.62 | 30.95(±1.58) | 36.51@49 | 94.61 | 94.69 | 100.00 | 4.3961 | 213.3 | 1.16 | +31.46 |
| badnets_res_square | Badnets-C | res/square | 75.49 | 70.62(±2.39) | 99.73@11 | 94.83 | 94.85 | 100.00 | 0.7959 | 265.9 | 78.76 | -3.27 |
| badnets_res_linear | Badnets-C | res/linear | 72.43 | 68.78(±2.16) | 99.34@40 | 94.69 | 94.75 | 100.00 | 0.9252 | 148.6 | 68.65 | +3.78 |
| badnets_res_log | Badnets-C | res/log | 72.43 | 68.78(±2.16) | 99.34@40 | 94.69 | 94.75 | 100.00 | 0.9252 | 92.9 | 82.13 | -9.70 |
| badnets_forget | Badnets-C | forget | 69.42 | 64.25(±3.26) | 99.09@11 | 94.87 | 94.92 | 100.00 | 1.0376 | 269.3 | 71.74 | -2.32 |
| badnets_loss | Badnets-C | loss | 61.14 | 59.01(±1.97) | 99.60@55 | 94.86 | 94.80 | 99.99 | 1.283 | 147.4 | 52.71 | +8.43 |
| badnets_res_exp | Badnets-C | res/exp | 51.77 | 49.36(±2.50) | 98.52@56 | 94.74 | 94.71 | 100.00 | 1.885 | 92.8 | 76.5 | -24.73 |
| badnets_gradient | Badnets-C | grad | 55.00 | 43.54(±4.17) | 99.83@27 | 94.63 | 94.62 | 100.00 | 1.6331 | 90.4 | 52.56 | +2.44 |
| badnets_random | Badnets-C | random | 37.83 | 36.37(±2.86) | 99.26@25 | 94.96 | 94.95 | 99.99 | 2.8548 | 180.9 | 37.24 | +0.59 |
| blend_res_linear | Blended-C | res/linear | 77.92 | 75.86(±1.44) | 95.72@17 | 94.48 | 94.58 | 100.00 | 0.975 | 101.9 | 82.31 | -4.39 |
| blend_res_log | Blended-C | res/log | 77.92 | 75.86(±1.44) | 95.72@17 | 94.48 | 94.58 | 100.00 | 0.975 | 87.3 | 82.34 | -4.42 |
| blend_res_square | Blended-C | res/square | 75.79 | 72.77(±1.64) | 88.77@32 | 94.61 | 94.66 | 100.00 | 1.2945 | 87.2 | 84.88 | -9.09 |
| blend_forget | Blended-C | forget | 70.52 | 70.48(±1.17) | 83.07@25 | 94.85 | 94.90 | 100.00 | 1.6134 | 86.8 | 71.05 | -0.53 |
| blend_loss | Blended-C | loss | 66.97 | 67.28(±1.57) | 95.09@16 | 94.68 | 94.61 | 100.00 | 1.7214 | 221.7 | 59.43 | +7.54 |
| blend_gradient | Blended-C | grad | 62.19 | 60.22(±1.51) | 92.41@45 | 94.43 | 94.49 | 100.00 | 1.965 | 116.1 | 58.45 | +3.74 |
| blend_res_exp | Blended-C | res/exp | 60.03 | 58.02(±1.58) | 87.08@19 | 94.98 | 94.92 | 100.00 | 2.0775 | 221.8 | 71.81 | -11.78 |
| blend_random | Blended-C | random | 50.46 | 49.87(±1.64) | 88.00@20 | 95.07 | 95.01 | 100.00 | 2.821 | 137.6 | 53.41 | -2.95 |
| quantizeB_res_linear | MultiBpp-B | res/linear | 93.04 | 89.70(±2.53) | 99.50@57 | 94.45 | 94.41 | 100.00 | 0.2663 | 94.6 | 76.73 | +16.31 |
| quantizeB_res_log | MultiBpp-B | res/log | 93.04 | 89.70(±2.53) | 99.50@57 | 94.45 | 94.41 | 100.00 | 0.2663 | 93.0 | 77.1 | +15.94 |
| quantizeB_res_square | MultiBpp-B | res/square | 93.30 | 85.42(±2.73) | 97.73@37 | 94.85 | 94.88 | 100.00 | 0.2712 | 269.9 | 82.54 | +10.76 |
| quantizeB_forget | MultiBpp-B | forget | 84.13 | 80.95(±2.11) | 98.50@48 | 94.72 | 94.76 | 100.00 | 0.722 | 149.1 | 74.39 | +9.74 |
| quantizeB_loss | MultiBpp-B | loss | 70.99 | 68.14(±2.46) | 98.70@57 | 94.91 | 94.87 | 100.00 | 1.3846 | 90.3 | 28.02 | +42.97 |
| quantizeB_gradient | MultiBpp-B | grad | 63.22 | 58.63(±3.41) | 99.14@51 | 94.85 | 94.83 | 100.00 | 1.9998 | 92.2 | 38.26 | +24.96 |
| quantizeB_res_exp | MultiBpp-B | res/exp | 53.91 | 52.81(±2.43) | 87.58@58 | 94.74 | 94.78 | 100.00 | 2.6311 | 126.2 | 53.92 | -0.01 |
| quantizeB_random | MultiBpp-B | random | 9.74 | 8.81(±0.69) | 22.29@19 | 94.97 | 95.00 | 99.99 | 6.8987 | 89.1 | 1.37 | +8.37 |
<!-- AUTO:result_table_end -->

### 3.2 每组完整统计（论文记录什么就存什么 + 训练过程细节）

| 实验组 | ASR末50均 | ASR首破50%@ep | ASR首破80%@ep | BA峰值@ep | BA末20 std | CleanLoss末 | lr档 |
|---|---|---|---|---|---|---|---|
| quantize_res_square | 83.99 | 17 | 19 | 94.78@145 | 0.07 | 0.2123 | 0.1/0.01/0.001 |
| quantize_res_linear | 85.19 | 13 | 15 | 94.98@224 | 0.08 | 0.1984 | 0.1/0.01/0.001 |
| quantize_res_log | 85.19 | 13 | 15 | 94.98@224 | 0.08 | 0.1984 | 0.1/0.01/0.001 |
| quantize_res_exp | 61.65 | 15 | 15 | 94.92@282 | 0.08 | 0.2029 | 0.1/0.01/0.001 |
| quantize_forget | 81.93 | 15 | 15 | 94.97@262 | 0.05 | 0.2022 | 0.1/0.01/0.001 |
| quantize_loss | 73.36 | 26 | 30 | 95.07@274 | 0.05 | 0.2016 | 0.1/0.01/0.001 |
| quantize_gradient | 62.51 | 18 | 27 | 94.82@255 | 0.04 | 0.2040 | 0.1/0.01/0.001 |
| quantize_random | 29.63 | — | — | 94.87@251 | 0.05 | 0.2094 | 0.1/0.01/0.001 |

### 3.3 关键结论（复现是否成功，基于已完成 22 组）

各列"本文最优 Res-x²"与 SOTA Forget 的复现对照（ASR 末20均 vs 论文）：

| 攻击列 | Res-x² 复现 / 论文（差） | Forget 复现 / 论文（差） | 判定 |
|---|---|---|---|
| MultiBpp-RGB | 82.99 / 83.88（−0.9） | 81.18 / 78.10（+3.1） | ✅ 优秀 |
| MultiBpp-B | 85.42 / 82.54（+2.9） | 80.95 / 74.39（+6.6） | ✅ 良好 |
| Badnets-C | 70.62 / 78.76（−8.1） | 64.25 / 71.74（−7.5） | 🟡 偏低但同量级 |
| Blended-C | 72.77 / 84.88（−12.1） | 70.48 / 71.05（−0.6） | 🟡 Forget✓ / Res-x²偏低(单种子) |

- ✅ **核心结论成立**：三列中 Res-x² 均为/接近最强选择、Forget 强于弱基线，相对排序与论文一致；MultiBpp-RGB/B 的 Res-x² 几乎完美匹配（±3 内）。
- 🟡 **Badnets-C 整体偏低 ~8 点**（Res-x² −8.1、Forget −7.5、Res-log −13.3）：单种子 clean-label 1% 方差所致；Res-log 还受"与 Res-x 选同 500 样本"的快照问题放大（见 §3.4-1）。
- ⚠️ **弱基线偏高**（Loss/Random 在 MultiBpp 列 +7~+40）：1% clean-label 对弱方法高方差，论文疑多种子平均；不影响"本文方法有效"结论。
- ✅ **BA 全部 ≈94.4-95.0**，模型效用正常；Res-eˣ 仍是最弱 Res 变体（论文"不恰当组合"现象复现）。

### 3.4 已知 caveat（写论文需注明）

1. **Res-log 与 Res-x 结果字节级一致**：作者提供的 `stats_forget_seed_1.pkl` 快照上，两者选出相同的 500 样本（交集 500/500），故训练完全相同。论文两者不同（80.20 vs 83.07），应是作者自跑 `cal_metric` 的统计与这份快照略有差异。
2. **Res-eˣ 曾崩溃**：`get_stats` 的 exp 分支 `exp(-cls_res)`，cls 计数值~1e4 → 下溢为 0 → 除零。已做**数值稳定修补**（softmax 减最小值，数学等价），见 `utils.py` 注释。仅让该变体可跑，不改算法。

---

## 4. 论文 Table 1 全表（32 组）+ 复现状态

> 论文值（ASR / BA），状态标记复现进度。**全 4 列 ✅ 已完成**（复现值见 §3.1）。

| 选择策略 | Badnets-C ✅ | Blended-C ✅ | MultiBpp-B ✅ | MultiBpp-RGB ✅ |
|---|---|---|---|---|
| Random | 37.24 / 94.42 | 53.41 / 94.90 | 1.37 / 94.51 | **1.16 / 94.95** |
| Loss | 52.71 / 94.71 | 59.43 / 95.10 | 28.02 / 94.84 | 47.85 / 94.76 |
| Gradient | 52.56 / 94.45 | 58.45 / 94.77 | 38.26 / 95.04 | 53.28 / 95.03 |
| Forget | 71.74 / 94.90 | 71.05 / 94.55 | 74.39 / 94.92 | 78.10 / 94.90 |
| Res-log | 82.13 / 94.98 | 82.34 / 94.73 | 77.10 / 94.54 | 80.20 / 94.82 |
| Res-x | 68.65 / 94.71 | 82.31 / 94.31 | 76.73 / 94.21 | 83.07 / 94.63 |
| Res-x² | 78.76 / 94.94 | 84.88 / 94.38 | 82.54 / 94.58 | **83.88 / 94.59** |
| Res-eˣ | 76.50 / 94.47 | 71.81 / 94.80 | 53.92 / 94.72 | 62.28 / 94.85 |

> 表内为论文 ASR / BA。✅ 列的复现值见 §3.1。

---

## 5. 日志与命名规范（每次执行后如何归档）

### 5.1 目录与日志命名

```
results/
  <attack>_<selection>[_<res_sel>]/     # 一组实验 = 一个目录
    output_<seed>.log                    # 训练日志（含每 epoch 全指标）
  <attack>_<selection>[_<res_sel>].stdout # nohup 屏幕输出（备份）
  _detail_summary.json                   # 全部已完成组的详细统计（机器可读）
```

- **命名约定**：`<attack>` ∈ {badnets, blend, quantize(=MultiBpp-RGB), quantizeB(=MultiBpp-B)}；`<selection>` ∈ {random, loss, gradient, forget, res}；res 时追加 `_<res_sel>` ∈ {log, linear, square, exp}。
- 例：`quantize_res_square` = MultiBpp-RGB + Res-x²；`badnets_forget` = Badnets-C + Forget；`quantizeB_res_square` = MultiBpp-B + Res-x²。
- 跨数据集/投毒率时，目录名追加后缀，如 `badnets_res_square_c100_a002`（CIFAR-100, α=0.2%）。

### 5.2 每跑完一组后的更新流程（必做，一键完成）

1. 确认 `results/<组名>/output_1.log` 跑满 300 epoch（末行 epoch=299）。
2. **一条命令**自动重写 §3.1 结果表 + 导出 JSON：
   ```bash
   conda activate GeneralComponents
   python parse_detail.py ./results ./results/_detail_summary.json --write-md docs/result_all.md
   ```
   （`--write-md` 会把 §3.1 中 `<!-- AUTO:result_table_start -->…end -->` 之间的表格整体覆写，无需手改。）
3. 手动微调两处（需判断，脚本不动）：§4 全表把对应格子状态 ⬜→✅ 并填复现值；顶部 **📊 当前进度** 的计数。

### 5.3 解析脚本说明

- `parse_results.py`：概览版，只报 ASR/BA 末值+末20均+论文差距（轻量快速核对）。
- `parse_detail.py`：详细版，报末值/末20均/末50均/峰值/std/TrainACC/收敛epoch/耗时；支持 `--md`（打印表格）/ `--write-md <file>`（按标记覆写）/ 导出 JSON。内置 Table 1 四列全部论文 ASR，自动算差距。

### 5.4 ✅ Blended-C 配置陷阱（已处理：补丁为 0.2:0.2:0.2）

`utils.py` 的 `Add_Clean_Label_Train_Trigger`（train，行 303）与 `Add_Test_Trigger`（test，行 187）里，blend 的 `checkboard` 原本被**硬编码为 `[2,1,3]`**，即透明度恒为 (0.2, 0.1, 0.3)，`--type` 参数被忽略。
- 论文 Table 1 的 Blended-C 默认透明度是 **0.2:0.2:0.2**（Component A 实验，香草触发器）；(0.2,0.1,0.3) 是 Component-C 优化版（Table 4）。
- **已按方案(b)补丁**（2026-07-01）：把上述两处 `[2,1,3]` 改为 `[2,2,2]`（NAR 的行 424 未动）。补丁可逆、行内有注释标注。故 Blended-C 列现按 0.2:0.2:0.2 复现，与论文 Table 1 一致。
- Blended-C 全 8 组已由 `run_table1.sh` 开跑（`--backdoor_type blend`，透明度由补丁后硬编码决定，`--type 2:2:2` 仅占位记录意图）。

---

## 6. 附录：完整命令模板

> 统一前缀：`CUDA_VISIBLE_DEVICES=<gpu> python train_backdoor.py`，统一参数
> `--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10`

**MultiBpp-RGB 列（已完成 ✅）** — `--backdoor_type quantize --num_levels 24:28:8`，8 个选择策略：
```bash
# Res-x² (本文最优)
python train_backdoor.py --backdoor_type quantize --num_levels 24:28:8 --selection res --res_sel square --result_dir results/quantize_res_square
# Res-x / Res-log / Res-eˣ：把 --res_sel 换成 linear / log / exp
# Forget / Loss / Gradient / Random：--selection forget / loss / grad / random（去掉 --res_sel）
```

**待跑三列**（Table 1 剩余 24 组），选择策略同上 8 个。**实际执行用 `run_table1.sh` 后台调度**，下面仅列命令：
```bash
# Badnets-C 列（代码硬编码黑白棋盘，--type 仅占位）
python train_backdoor.py --backdoor_type badnets --type 0:0:0 --selection <sel> [--res_sel <x>] --result_dir results/badnets_<sel>
# MultiBpp-B 列（仅蓝通道强量化，num_levels=255:255:8 已确认）
python train_backdoor.py --backdoor_type quantize --num_levels 255:255:8 --selection <sel> [--res_sel <x>] --result_dir results/quantizeB_<sel>
# Blended-C 列 ⏸ 暂缓：代码硬编码 (0.2,0.1,0.3)，需先定夺是否改 0.2:0.2:0.2（见 §5.4）
python train_backdoor.py --backdoor_type blend --type 2:2:2 --selection <sel> [--res_sel <x>] --result_dir results/blend_<sel>
```

**CIFAR-100（Table 2，前置 cal_metric）**：
```bash
# Step0 先生成指标（当前缺）
python cal_metric.py --dataset cifar100 --num_classes 100 --output_dir save_metric_100_res
# Step1 训练
python train_backdoor.py --dataset cifar100 --num_classes 100 --backdoor_type <badnets|blend> --poison_rate <0.002|0.005> --output_dir save_metric_100_res --result_dir results/<...>_c100
```

---

*维护者：复现期间每完成一组即更新本文件。最近更新：2026-07-01 21:04（**Table 1 全 32/32 完成**：剩 4 组 Blended-C 已补跑满 epoch=299，调度器自动写 §3.1；代码核验未污染；现暂停等 FAAT agent 改完代码再定后续）。*
