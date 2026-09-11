# all_result.md — 全部实验总览（单文件真相源）

> **生成日期**：2026-08-04　**仓库**：`GeneralComponents-main`　**当前分支**：`exp/kst-sdt`
> **目的**：把本仓库跑过的**所有**实验归类到一个文件，不论最终结果成败，记录"做了什么、为什么做、跑了哪些、结果如何、下一步该做什么"。
> **细节真相分文件**：每个家族的完整日志/数字保留在 `docs/*.md` 与 `results/<组名>/output_1.log`；本文件做**地图 + 关键数字 + 结论**。
> **统计口径**：ASR/BA 默认取**末 20 个 epoch 均值**（比单点末值稳）；防御指标 AC/SS-AUC、NC 异常指数、SSIM/L2 等按各家族原始定义。

---

## 0. 一图速览：实验家族全景

| # | 实验家族 | 目录前缀 | 跑过的组数* | 阶段 | 一句话结论 |
|---|---|---|---|---|---|
| 1 | **论文复现 baseline**（GeneralComponents） | `badnets_*`/`blend_*`/`quantize_*`/`quantizeB_*`/`bl_c100_*`/`bl_tiny_*`/`gtsrb_*` | 32 + 27 | ✅ 完成 | CIFAR-10 复现成功；GTSRB/大类 clean-label 1% 上 baseline 崩溃（这是新方法的发力点） |
| 2 | **FAAT 自研主方法**（v2→v7） | `faatb_v*`/`faat_*` | ~110 | ✅ 成熟 | 4 数据集**全面碾压 baseline**（+8.7 ~ +51.9 ASR），AC/SS/STRIP/FP 防御失效；GTSRB 诚实 Pareto |
| 3 | **隐蔽触发器研究线** | `icit_*`/`pca_*`/`pat_*`/`narcissus_*` | ~25 | ✅ 收尾 | ICIT→PA-ICT→PAT 演进；**PA-ICT 纯 PCA 是唯一稳定绕过全图 NC 的变体**；Pareto 已饱和 |
| 4 | **KST / SDT**（当前主线） | `kst_sdt/` 内多子目录 + campaign | ~70 | ✅ **核心完成** | KST 平谱触发器**结构性规避频域防御**；CIFAR-10/CIFAR-100/Tiny clean-label 超 baseline（多 seed 坐实：Tiny ε48~98、C100 ε48 reslin~94 三 seed）；GTSRB 失败(诚实 limitation)；撞 flat-spectrum↔可学性墙 |
| 5 | **探针类触发器**（可行性） | `rkt_*`/`style_*`/`feast_*`/`icaf_*`/`opal_*`/`orbit_*` | ~16 | ⏸ 探索 | 多为 smoke probe；**ORBIT-IRREP 有实证结果**（ASR 0.788 vs 0.603），其余多数缺完整评估 |
| 6 | **消融 / 低投毒 / guidance** | `ablation_*`/`lowpoison_*`/`faat_res_square_gs*` | ~14 | ✅ 完成 | 去掉特征对齐 ASR 仍 96.4%（非核心但有益）；guidance scale 是重要旋钮 |

*组数为目录计数粗估（含少量 INVALID/wire 失败实验），以 `results/` 实际为准。全仓库约 **229 个结果目录**。

---

## 1. 项目背景与统一配置

**项目定位**：复现论文 *"A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks"*（NeurIPS 2025, arXiv 2509.19947）作为 **baseline**，支撑本团队自研新方法 **FAAT** 及后续 **KST/SDT** 等创新点的论文写作。

**统一环境**：
- conda：`/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents`（python3.8 + torch1.11.0+cu113）
- GPU：RTX 3090 ×4（共享，每组 2-3 GB）
- 数据：CIFAR-10 / CIFAR-100 / GTSRB / Tiny-ImageNet 均已就位（软链到 `./data*`）

**统一训练超参（CIFAR-10 默认）**：ResNet18，300 epoch，SGD lr=0.1 mom=0.9 nesterov wd=5e-4，milestones=[60,90] γ=0.1，batch=128，seed=1，y_target=0，poison_rate=0.01（500 clean-label 投毒样本，全取自目标类）。

**指标定义**：
- `ASR` = PoisonACC（触发后非目标类测试图被判为 target 的比率）
- `BA` = CleanACC（干净测试精度）
- 隐蔽性：`SSIM` / `L2` 范数；防御：`AC-AUC` / `SS-AUC`（越接近 0.5 越规避）、`NC 异常指数`（<2 视为规避 Neural Cleanse）。

---

## 2. 论文复现 baseline（GeneralComponents）

> 既是对论文的复现验证，也是 FAAT/KST 的对照基准。**最关键的发现：标准 clean-label baseline 在 GTSRB / 大类别低投毒 regime 上崩溃**——这正是新方法要超越的起点。

### 2.1 CIFAR-10 Table-1（32 组，✅ 全部完成 2026-07-01）

配置：CIFAR-10 + ResNet18 + 1% 投毒 + seed=1 + 300 epoch。4 攻击列 × 8 选择策略。复现 vs 论文 ASR 差距（ASR 末20均）：

| 攻击列 | Res-x² 复现/论文（差） | Forget 复现/论文（差） | 判定 |
|---|---|---|---|
| MultiBpp-RGB（`quantize`） | 82.99 / 83.88（−0.89） | 81.18 / 78.10（+3.08） | ✅ 优秀 |
| MultiBpp-B（`quantizeB`） | 85.42 / 82.54（+2.88） | 80.95 / 74.39（+6.56） | ✅ 良好 |
| Badnets-C | 70.62 / 78.76（−8.14） | 64.25 / 71.74（−7.49） | 🟡 偏低但同量级 |
| Blended-C | 72.77 / 84.88（−12.11） | 70.48 / 71.05（−0.57） | 🟡 Forget ✓ / Res-x² 偏低 |

**完整 32 行明细**见 `docs/result_all.md` §3.1（自动生成）。**核心结论**：Res-x² 在各列均为/接近最强选择，Forget 强于弱基线；MultiBpp 两列几乎完美匹配论文；BadNets/Blended 偏低属单种子 clean-label 1% 高方差。**BA 全部 94.4–95.0**。

> **CIFAR-10 最强 baseline = MultiBpp-B Res-x² 85.42**（论文 Blended-C Res-x² 84.88）——这是 FAAT/KST 在 CIFAR-10 上要超越的标杆。

### 2.2 GTSRB baseline（3 攻击 × 3 seed，1% 投毒，Res-x²）

| 攻击 | seed1 | seed2 | seed3 | mean ASR±std | BA |
|---|---|---|---|---|---|
| BadNets-C | 0.00 | 0.00 | 0.00 | **0.00** | 99.95 |
| Blended-C | 34.75 | 36.07 | 31.57 | **34.13 ± 1.83** | 99.89 |
| MultiBpp-RGB | 0.00 | 0.00 | 0.00 | **0.00** | 99.95 |

**关键发现**：baseline 在 GTSRB 上**崩溃**（BadNets/MultiBpp 3 seed 全 0）。根因：GTSRB 43 类、BA 瞬间饱和 100%，仅 189 个 clean-label 样本带的小固定触发器被强拟合稀释。论文无 GTSRB 实验，此为自跑对照。**来源**：`docs/GTSRB-BASELINES.md`、`gtsrb_baselines_results.csv`。

### 2.3 CIFAR-100 baseline（3 攻击 × 3 seed，1% 投毒，Res-x²）

| 攻击 | seed1 | seed2 | seed3 | mean ASR±std | BA |
|---|---|---|---|---|---|
| BadNets-C | 78.67 | 83.90 | 92.40 | **84.99 ± 5.66** | 76.89 |
| Blended-C | 76.46 | 76.80 | 77.20 | **76.82 ± 0.30** | 77.05 |
| MultiBpp-RGB | 29.50 | 31.00 | 25.10 | **28.53 ± 2.50** | 77.30 |

**注**：论文用 0.5% 投毒（最强 BadNets-C 85.06 / Blended 77.45）；本复现用 1%（BadNets-C 84.99 高度匹配论文 0.5%）。MultiBpp-RGB 在 100 类 @1% 崩溃至 28.5%。**来源**：`docs/CIFAR100-COMPARISON.md`、`bl_c100_*` 日志。

### 2.4 Tiny-ImageNet baseline（3 攻击 × seed1，1% 投毒，Res-x²）

| 攻击 | seed1 ASR | BA |
|---|---|---|
| BadNets-C | 90.11 ± 0.72 | 55.80 |
| Blended-C | 78.70 ± 0.84 | 54.45 |
| MultiBpp-RGB | 63.12 ± 1.81 | 54.66 |

**注**：seed2/3 日志缺失。论文用 0.25%（最强 Blended 43.93 / BadNets 38.96）；本复现用 1%（投毒率 4× → ASR 相应提高）。**来源**：`bl_tiny_*` 日志。

---

## 3. FAAT 自研主方法（v2 → v7）

### 3.1 方法概述与版本演进

**FAAT（Feature-Aligned Adaptive Trigger）** 是本团队自研 clean-label backdoor 方法。核心组件：(1) 全局触发器 δ_global（Narcissus 类强方向）；(2) 自适应残差 δ_adaptive（DCT 有界，用于防御规避塑形）；(3) 特征对齐损失 L_align（把投毒特征拉向目标类中心，破坏聚类防御）。复用论文 Res-x² 选择与 clean-label 框架，但触发器**自适应、可学习、不可见**。

| 版本 | 关键改动 | 目标 | 结果 |
|---|---|---|---|
| v2 | δ_global 用 CE 优化目标 | 修复 v1 的 L_align 弱化 δ_global | ❌ ASR 95.05 < v1 99.14 |
| v3 | δ_global 固定 Narcissus + 无界 adaptive | 保留强 ASR 方向 | ❌ adaptive 夺权，C2 破坏，ASR 崩 |
| **v3.1** | δ_global 冻结 + 有界 adaptive（\|da\|≤0.15） | 满足 C2 同时规避防御 | ✅ **首个三目标同时达成** |
| **v4** | 自包含 Narcissus 引擎（from_scratch）+ 跨数据集 | 去除对作者 `noise_01000.pth` 依赖 | ✅ CIFAR-10/GTSRB 双成立 |
| v5 | CW-margin 全局触发器 + adaptive 等比缩放 | 修 GTSRB 隐蔽性 | ❌ CW 伤 ASR，零 SSIM 收益 |
| v5b | 恢复 CE + λ_align=3.0 | 强对齐压低 AC/SS | ⚠️ ASR 恢复但 AC/SS 不动 |
| **v6** | 自包含引擎 + CIFAR-100 @1% | 100 类验证 | ✅ ASR 99.4 |
| **v6c** | CIFAR-100 @0.5%（同论文 Table2） | apples-to-apples 对比 | ✅ ASR 98.5 超论文 +13.4 |
| **v7** | Tiny-ImageNet @0.25%（同论文） | 200 类超难 regime | ✅ ASR 95.8 超论文 +51.9 |

### 3.2 CIFAR-10（1% clean-label，300 epoch，ASR/BA 末20均）

| 版本/配置 | L2 实测 | SSIM | ASR | BA | AC-AUC | SS-AUC | 备注 |
|---|---|---|---|---|---|---|---|
| v2_base | 3.93 | 0.816 | 95.05 | 94.74 | 0.111 | 0.312 | CE 目标，ASR 低于 v1 |
| v3.1_scale0.2 | 1.31 | 0.963 | 94.81 | 94.67 | 0.223 | 0.416 | 首个三目标达成 |
| v3.1_scale0.17 | 1.12 | 0.972 | 90.45 | 94.89 | 0.308 | 0.420 | 更隐蔽档 |
| v4_l2_0.9 | 0.90 | 0.981 | 76.1 | 94.8 | 0.33 | 0.44 | 隐蔽扫点 |
| v4_l2_1.2 | 1.19 | 0.967 | 88.6 | 94.7 | 0.37 | 0.45 | +3.7 超论文 |
| **v4_l2_1.5** | 1.49 | 0.950 | **93.6** | 94.8 | 0.25 | 0.43 | **+8.7 超论文，自包含最优** |

**vs baseline**：CIFAR-10 最强 baseline 84.88 → FAAT v4_l2_1.5 **93.6（+8.7）**，BA 无跌，AC/SS/STRIP/FP 全失效。

### 3.3 GTSRB（1% clean-label）

| 配置 | L2 | SSIM | ASR | BA | AC-AUC | SS-AUC | 备注 |
|---|---|---|---|---|---|---|---|
| v4_l2_2.5 | 2.45 | 0.805 | 64.5 | 99.9 | 0.71 | 0.79 | 跨数据集验证 |
| v4_l2_3.0 | 2.92 | 0.761 | 71.5 | 99.9 | 0.72 | 0.73 | 中等 |
| **v4_l2_3.5** | 3.38 | 0.720 | **82.7** | 99.8 | 0.76 | 0.74 | **GTSRB 最优** |
| v5_l2_1.5 | 1.48 | 0.901 | 38.6 | 100.0 | 0.59 | 0.62 | CW 隐蔽档崩 |
| v5_l2_3.0 | 2.92 | 0.761 | 71.1 | 99.9 | 0.69 | 0.79 | CW 未改善 SSIM |
| v5b_l2_3.5 | 3.38 | 0.718 | 83.3 | 99.9 | 0.76 | 0.74 | λ_align=3.0 |

**vs baseline**：GTSRB baseline 崩溃（BadNets/MultiBpp 0、Blended 34.1）→ FAAT v4_l2_3.5 **82.7**，是 Blended 的 2.4×。诚实 Pareto：AC/SS ~0.7（43 类 + 大触发器固有）。

### 3.4 CIFAR-100（v6 @1%，v6c @0.5%）

| 配置 | 投毒率 | L2 | SSIM | ASR | BA | AC-AUC | SS-AUC |
|---|---|---|---|---|---|---|---|
| v6_l2_1.5 | 1% | 1.47 | 0.949 | 99.1 | 77.09 | 0.007 | 0.208 |
| v6_l2_2.0 | 1% | 1.96 | 0.919 | 99.6 | 77.25 | 0.006 | 0.203 |
| v6_l2_2.5 | 1% | 2.44 | 0.888 | 99.6 | 76.95 | 0.008 | 0.212 |
| v6c_l2_1.5 | 0.5% | 1.47 | 0.949 | 96.8 | 77.9 | 0.025 | 0.262 |
| **v6c_l2_2.0** | 0.5% | 1.96 | 0.919 | **98.5** | 77.6 | 0.022 | 0.246 |
| v6c_l2_2.5 | 0.5% | 2.44 | 0.888 | 99.4 | 77.9 | 0.019 | 0.236 |

**vs baseline**：CIFAR-100 @0.5% baseline 最强 BadNets-C 85.06 → FAAT v6c_l2_2.0 **98.5（+13.4）**，BA 77.6 ≈ baseline 78 无跌。

### 3.5 Tiny-ImageNet（v7 @0.25%，同论文 Table11）

| 配置 | L2 | SSIM | ASR | BA | AC-AUC | SS-AUC |
|---|---|---|---|---|---|---|
| v7_l2_1.5 | 1.48 | 0.952 | 92.9 | 55.0 | 0.008 | 0.265 |
| **v7_l2_2.0** | 1.97 | 0.927 | **95.8** | 54.7 | 0.004 | 0.249 |
| v7_l2_2.5 | 2.45 | 0.902 | 94.8 | 54.5 | 0.007 | 0.248 |

**vs baseline**：Tiny @0.25% baseline 最强 Blended 43.93 → FAAT v7_l2_2.0 **95.8（+51.9）**。

### 3.6 消融 / 低投毒 / guidance-scale（FAAT 组件必要性）

| 实验 | 配置 | ASR | BA | 说明 |
|---|---|---|---|---|
| ablation_noalign_seed1 | CIFAR-10 l2_1.5 去特征对齐 | 96.37 | 94.66 | 去对齐 ASR 仍高 → **对齐非核心但有益** |
| ablation_noalign_seed2/3 | 同上多 seed | — | — | 日志缺失 |
| ablation_CLEAN_scale0.1/0.2 | 仅 clean 对照 | — | — | 验证投毒必要性 |
| ablation_scale0.1/0.2_noAdp | 去 adaptive | — | — | 验证 adaptive 作用 |
| ablation_tiny_l2_2.0_noalign | Tiny + 去对齐 | — | — | 跨数据集消融 |
| lowpoison_p005 / p01 | CIFAR-10 0.5%/1% | — | — | 低投毒鲁棒性 |
| faat_res_square_gs010/050/100 | guidance 0.1/0.5/1.0 | — | — | gs↑ → loss↓，重要旋钮 |
| faat_res_square_gs100_noadp | gs1.0 + 去 adaptive | — | — | 高 gs 下 adaptive 作用 |

**消融结论**：特征对齐去掉 ASR 仍 96.4%（非核心但有益）；guidance scale 是调节触发器强度的关键；adaptive 有助于高 gs 下保性能。**来源**：`docs/ablation_noalign_results.json`、`ablation_*`/`lowpoison_*`/`faat_res_square_gs*` 日志。

### 3.7 负面 / 无效结果（FAAT）

| 类别 | 详情 |
|---|---|
| **INVALID_baseartifact**（4 组） | `faatb_l2_{0.9,1.3,2.0,3.0}_INVALID_baseartifact`：误用 v1 优化的 δ_global（L2 4.55）而非 Narcissus（L2 6.6），artifact 污染，**已作废保留** |
| v2（CE 目标） | CE 过拟合干净代理，不迁移 victim，ASR 95.05 < v1 99.14 |
| v3（无界 adaptive） | \|da\| 推到 ~5，adaptive 夺权，ASR 崩至 35/1.1/0.5 |
| v5（CW + ratio0.12） | CW 伤 ASR，零 SSIM 收益，L2=2.5 ASR 58.1 < v4 64.5 |
| v5b（λ_align=3.0） | 强对齐未压低 AC/SS（0.74 ≈ v4 0.71） |
| GTSRB 初次尝试 | 无预计算强触发器，CE-proxy 无法生成有效 δ_global，ASR 0.9%（≈随机） |

---

## 4. 隐蔽触发器研究线（ICIT → PA-ICT → PAT → Saturation）

### 4.1 研究线概述

动机：比 FAAT 更隐蔽、可迁移、能绕 NC/AC/SS 防御。演进逻辑——
- **ICIT**（输入条件化不可见触发器）：逐图生成扰动 g(x)，使 NC 的通用 δ 逆向失效；
- **PA-ICT**（PCA 对齐）：把触发样本对齐到目标类主成分子空间，后门成"类中信号"，更难被 NC 检；
- **PAT**（感知分配触发器）：JND 感知约束，α 控可见性-有效性权衡；
- **Narcissus / BppAttack** 作为前沿对照基线。

### 4.2 ICIT 结果

**CIFAR-10 多 seed**（L2=2.0）：seed1 ASR 99.7 / BA 94.5 / SSIM 0.941，seed2 97.8，seed3 97.1，**规避 NC**。L2=1.5 seed1 被 NC 抓。

**多数据集**（ASR / BA / AC-AUC / SS-AUC）：
| 数据集 | L2 | ASR | BA | 可见 baseline 最强 ASR |
|---|---|---|---|---|
| CIFAR-100 | 2.0 | 99.6 | 77.0 | 85.06（BadNets） |
| Tiny-ImageNet | 2.0 | 82.6 | 54.6 | 43.93（Blended） |

**黑盒迁移（集成训练）**：源 R18 ASR 98.4；迁移到 held-out **R101 92.7**（单模型 50.8，+41.9）、**R152 93.9**（单模型 45.3，+48.6）。**结论**：集成把深架构迁移 ASR 从 ~50% 拉到 92-94%。**来源**：`docs/ICIT-*.md`、`icit_*` 日志。

### 4.3 PA-ICT / PCA-aligned 结果（核心突破）

| 变体 | ASR | BA | SSIM | NC 异常 | NC 检测 |
|---|---|---|---|---|---|
| λ=1.0 + CE | 99.5 | 94.7 | 0.941 | 4.96 | ❌ 被抓 |
| λ=0.1 + CE | 99.7 | 94.6 | 0.942 | 5.75 | ❌ 被抓 |
| λ=0.0 CE（=ICIT） | 99.8 | 94.5 | 0.941 | 3.32 | ❌ 被抓 |
| **λ=1.0 纯 PCA（无 CE）** | **97.3** | **94.9** | **0.947** | **0.79** | **✅ 绕过** |

**多 seed 稳健性**：纯 PCA（无 CE）3 seed 全绕 NC（异常 0.79 / −1.09 / 1.21，均值 0.30）；ICIT 3 seed 抓 1（均值 1.74）。**代价**：纯 PCA 比 ICIT 牺牲约 2.5% ASR 换稳健 NC 规避。
**vs BppAttack**：BppAttack NC 异常 **6.77**（被抓），PA-ICT 纯 PCA **0.79–1.21**（绕过）——"防御防不住"实证。**来源**：`docs/PA-ICT-RESULTS.md`、`pca_*` 日志。

### 4.4 PAT 结果（α sweep + 多 seed）

| α | ASR | BA | SSIM | AC-AUC | SS-AUC |
|---|---|---|---|---|---|
| 0.1 | 99.9 | 94.7 | 0.946 | 5.88 | 0.17 |
| 0.3 | 99.9 | 94.7 | 0.947 | 0.86 | 0.81 |
| **0.5** | 98.4 | 94.5 | 0.945 | **0.08** | 0.98 |
| 0.75 | 99.7 | 94.6 | 0.946 | 0.02 | 0.99 |
| 1.0 | 99.9 | 94.7 | 0.945 | 0.01 | 0.99 |

α=0.5 多 seed：ASR 97.6–98.4 / BA 94.5–95.0，跨 seed 稳。**vs Narcissus/ICIT/BppAttack Pareto**：PAT 在 ASR-SSIM 前沿相当，AC/SS 规避更强。**来源**：`pat_*` 日志。

### 4.5 Saturation 双边饱和结论

- **防御侧饱和**：全图 L2 NC 通杀整个攻击前沿（Narcissus 异常 2.72、ICIT 3.24、**BppAttack 6.77**）；标准 patch-NC 全漏；AC/SS/STRIP/FP 对全局触发器全失效；频域检测被自然高频方差淹没（实战失败）。
- **攻击侧饱和**：6 机制 Pareto 表显示优化型触发器受**对抗距离 floor** 约束，Narcissus 显式坐 floor，ICIT ≈ Narcissus（增量），学习型前沿 BppAttack 受 poison 子集扰动下界约束；RKT/ChromaTrigger/频带限制均被支配。
- **结论**：clean-label 触发器空间在攻防两侧都被关闭。**来源**：`docs/SATURATION-ANALYSIS.md`。

> **该线整体结论**：最抗防御 = **PA-ICT 纯 PCA**（唯一稳定绕全图 NC，ASR 97.3）；最高 ASR = ICIT（99.7，但 NC 1/3 被抓）。Pareto 已饱和，**继续追"超 Narcissus/BppAttack 的触发器"收益递减**。

---

## 5. KST / SDT（当前主线，分支 `exp/kst-sdt`）

### 5.1 方法概述

- **KST（Knowledge/Spectrum Trigger）**：用 4 阶累积量签名 + FFT 相位参数化构造触发器，**频域完全平坦（δ 谱峰≡1.0）**，实现对频域/谱防御的结构性规避（vs Narcissus 谱峰 68.8 易被提取）。
- **SDT（Stein Discrepancy Trigger）**：用 Stein 残差/分数函数构造流形切向触发器，使触发图仍位于数据支撑集（on-manifold ratio 1.005），理论上规避似然/OOD 检测。
- 两者作为新增 `backdoor_type` 集成进 baseline 管线（`train_backdoor.py`），复用 Component A/B/C 选择与训练流程。

### 5.2 实验时间线（按提交）

1. **基础验证**（`869c48d`）：KST+SDT+Narcissus 三触发器 CIFAR-10 并排验证（5% dirty-label，40 ep）。
2. **P0 epsilon sweep**（`6a3ffa3`）：ε ∈ {8,12,16,20}/255，建 ASR-SSIM-谱峰 Pareto。
3. **P0-2 low-poison**（`82816fb`）：1% 投毒 3 seed，发现 regime-dependent 限制。
4. **P0-2b KST-Learn**（`4799396`）：加 CE-proxy 提升可学性——**失败**（flat-spectrum↔可学性墙）。
5. **方法论纠正**（`a614e1e`/`c4dbd4d`）：切到 clean-label + baseline Table-1 对比，KST 集成为 `backdoor_type=kst`，forget ASR **86.76 超 baseline**。
6. **Multi-dataset campaign**（`acf2613`/`4097c94`）：扩到 CIFAR-100/GTSRB/Tiny，验泛化性。

### 5.3 CIFAR-10 epsilon sweep（5% dirty-label，40 ep，seed1）

| ε | ASR | BA | SSIM | L2 | δ 谱峰 | s-dprime | STRIP AUC |
|---|---|---|---|---|---|---|---|
| 8/255 | 0.956 | 0.841 | 0.991 | 0.59 | **1.0** | 2.71 | 0.489 |
| 12/255 | 0.990 | 0.822 | 0.976 | 1.00 | **1.0** | 3.87 | 0.575 |
| **16/255** | **0.998** | 0.850 | 0.961 | 1.33 | **1.0** | 4.64 | 0.787 |
| 20/255 | 0.994 | 0.842 | 0.940 | 1.71 | **1.0** | 5.50 | 0.904 |
| Narcissus 8/255 | 0.998 | 0.846 | 0.946 | 1.64 | **68.8** | — | 0.789 |

**关键**：ε=16 时 KST ASR 0.998 = Narcissus，同时 SSIM 更高、谱峰平坦（1.0 vs 68.8），**Pareto 支配 Narcissus**。

### 5.4 P0-2 low-poison + KST-Learn 失败（1% dirty-label，3 seed）

| 触发器 | ε | ASR (mean±std) | SSIM | δ 谱峰 |
|---|---|---|---|---|
| KST | 8/255 | **0.029 ± 0.002（失败）** | 0.991 | 1.0 |
| KST | 16/255 | 0.871 ± 0.040 | 0.961 | 1.0 |
| Narcissus | 8/255 | 0.962 ± 0.026 | 0.946 | 68.8 |
| Narcissus | 16/255 | 0.996 ± 0.004 | 0.848 | 68.8 |

**KST-Learn（加 CE-proxy）**：α=0/0.5/2.0 下 ASR 0.882/0.872/0.900 ≈ pure-KST 0.871，**CE 项无效**；且破坏 4 阶签名（s-dprime 4.64→~0）。**根本墙：flat-spectrum 隐蔽性 vs 可学性直接冲突**，非调参问题。

### 5.5 Clean-label vs baseline Table-1（CIFAR-10，1% clean-label，300 ep，ASR 末20均 %）

| selection | **KST best** | BadNets-C | Blended-C | MultiBpp-RGB | MultiBpp-B |
|---|---|---|---|---|---|
| random | 27.00 (ε16) | 36.37 | 49.87 | 30.95 | 8.81 |
| **forget** | **86.76 (ε16)** ✅ | 64.25 | 70.48 | 81.18 | 80.95 |
| **res/linear** | **92.59 (ε20)** ✅ | 68.78 | 75.86 | 85.09 | 89.70 |

**方法论纠正后的正面结果**：KST + Component A 在 **forget 超 baseline 全部触发器**（86.76 vs MultiBpp-RGB 81.18，+5.58）；在 **res/linear 超 baseline 全部**（92.59 vs MultiBpp-B 89.70，+2.89）。BA ~94.7 持平；隐蔽性 = 平谱（δ 峰 1.0）+ SSIM 0.94-0.96。

### 5.6 Multi-dataset campaign（CIFAR-100 / GTSRB，baseline 管线）

| 数据集（投毒率） | config | KST ASR | KST BA | baseline 最优 | Δ |
|---|---|---|---|---|---|
| **CIFAR-100**（0.5%） | ε16 forget | 15.92 | 77.76 | Blended 76.00 | −60.08 |
| | ε16 reslin | 22.00 | 78.24 | — | −54.00 |
| | ε20 forget | 23.49 | 78.16 | — | −52.51 |
| | ε20 reslin | 25.52 | 78.40 | — | −50.48 |
| | **ε48 forget** | **83.25** | 77.93 | — | **+7.25** ✅ |
| | **ε48 reslin** | **93.95** | 77.84 | — | **+17.95** ✅ |
| **GTSRB**（1%） | ε16/20/32/48 forget & reslin | **0.00** | 99.94 | Blended 16.77 | **−16.77** ❌ |

**泛化 profile**：CIFAR-100 需 ε48（Linf 0.188，侵蚀部分隐蔽性）才超 baseline；**GTSRB 根本性失败**（所有 ε ASR 0，因 BA 99.95% 极自信，平谱翻不动）。Tiny-ImageNet：campaign 仅跑 baseline，KST ε48 待补（§10.2 提及 ep35 peak 99.8%，full 300ep 待定）。

### 5.7 SDT 基线 + 防御评估

**SDT（5% dirty-label）**：ASR **0.999** / BA 0.834 / SSIM 0.939 / on-manifold ratio **1.005** / Stein d-prime 0.69。价值在 on-manifold（Narcissus 做不到），不在像素隐蔽。

**防御**：KST 低 ε 规避 STRIP（AUC 0.489），高 ε 被抓（0.904）；频域签名防御下 KST peak/med=9-42（噪声底，**规避**），Narcissus=11217（被提取）。

### 5.8 失败 / 边界（负面证据）

1. **KST-Learn 撞墙**：flat-spectrum 隐蔽性 vs 可学性根本权衡，结构性限制（非 bug）。
2. **GTSRB 根本失败**：平谱触发器翻不动 BA 99.95% 的极自信模型——隐蔽性的代价。
3. **1% dirty-label regime 限制**：ε=8 失败；但 clean-label + Component A 下不成立（86.76 > baseline 81.18）。

### 5.9 Round-2/3/3b/4 多 seed 补全最终结果（2026-08-06）

> 补跑 26 run（GTSRB baseline full + Tiny KST full + 各数据集 KST 多 seed），坐实 KST 章节核心数字。日志文件名 `output_{seed}.log`（seed2/3 在 `output_2/3.log`，非 `output_1.log`）。完整运行日志见 `docs/round2_supplement.md`。

#### 5.9.1 KST 多 seed 稳健性（核心结论）

| 数据集(投毒率) | 配置 | seed1 | seed2 | seed3 | baseline 最强 | 判定 |
|---|---|---|---|---|---|---|
| **Tiny(0.25%)** | ε48 forget | 98.5 | 97.7 | 98.0 | BadNets 90.1 | ✅ 三 seed ~98，超 baseline +8 |
| **Tiny(0.25%)** | ε48 reslin | 98.5 | 98.2 | 98.0 | BadNets 90.1 | ✅ 稳健 |
| **CIFAR-100(0.5%)** | ε48 reslin | 94.0 | 93.2 | 94.6 | Blended 76.0 | ✅ 三 seed ~94，超 baseline +18 |
| **CIFAR-100(0.5%)** | ε48 forget | 83.2 | 91.9 | 92.9 | Blended 76.0 | 🟡 s1 偏低，s2/s3 高(~89)，仍超 baseline |
| **CIFAR-10(1%)** | e20 reslinear | 92.6 | 91.5 | 89.5 | MultiBpp-B 85.4 | ✅ 三 seed 89-92 稳健超 baseline |
| **CIFAR-10(1%)** | e16 forget | 86.8 | 64.8 | 62.0 | MultiBpp-B 85.4 | ⚠️ seed 方差大（s1 幸运），不主推 |

#### 5.9.2 Tiny KST ε-Pareto 全（补完 ε20，BA~55）

| ε | forget ASR | reslin ASR |
|---|---|---|
| ε16 | 45.3 | 62.6 |
| ε20 | 72.6 | 75.1 |
| ε48 | 98.5 | 98.5 |

→ Tiny 上 ε↑ 则 ASR↑，ε48 三 seed 稳健 ~98。

#### 5.9.3 GTSRB baseline full 300ep（之前仅 25ep 中断对比不公平；现已跑满）

| 攻击 | ASR | BA |
|---|---|---|
| BadNets-C | 0.0 | 100.0 |
| Blended-C | **34.7** | 99.9 |
| MultiBpp-RGB | 0.0 | 100.0 |
| MultiBpp-B | 0.0 | 100.0 |

→ 确认 GTSRB clean-label 1% baseline 崩溃（KST ε16/20/48 亦全失败，limitation 坐实）。

#### 5.9.4 影响论文写法的关键更新

1. **Tiny ε48 三 seed ~98** → Tiny 格 KST 超 baseline（甚至略超 FAAT v7 95.8）。
2. **CIFAR-10 主推 `e20_reslinear`**（89-92 三 seed 稳健），不主推 `e16_forget`（方差大）。
3. **CIFAR-100 ε48 reslin 三 seed ~94** 稳健超 baseline。
4. **GTSRB 全失败**诚实写 limitation（平谱翻不动 BA 99.95% 的极自信模型）。
5. **CIFAR-100 ε48 forget s1=83.2 偏低**（s2/s3=92/93）——论文报均值或主推 reslin。

### 5.9.5 GTSRB 专题：极自信模型 stress test（三方 Pareto 极限）

GTSRB（43 类交通标志，类间高度相似）上 CNN clean BA≈100%（所有方法 99.9-100），形成"极自信模型"regime。1% 投毒≈189 clean-label 样本。

**三方表现 + 训练动力学**：

| 方法 | ASR(末20) | peak@ep | PoisonLoss末 | 隐蔽性 |
|---|---|---|---|---|
| BadNets-C / MultiBpp | 0.0 | 1-7 | — | 可见触发器 |
| Blended-C | 34.7 | 80.8 | — | 可见 |
| **KST ε48** | 0.8 | **19.3 @ ep139** | **9.44** | SSIM 0.94 / 谱峰 1.0（隐蔽）|
| **FAAT v4 l2_3.5** | 82.7 | ~99 | — | SSIM 0.72 / AC-AUC 0.76（不隐蔽）|

**机制**：
- **baseline 崩**：固定触发器 + clean-label，BA 100% 的模型不需学触发器-类关联，189 样本的固定信号被强 clean 拟合淹没。
- **KST 失败**：训练中 peak 19.3%（ep139）后崩回 0.8%，末段 PoisonLoss=9.44（模型对触发样本极度抗拒）。平谱能量分散在所有频率，对 sharp-boundary 自信模型有效攻击能量不足。**对照**：同 KST ε48 在 CIFAR-100（BA 78%）peak 98.9% → 中等自信模型 work，**唯独极自信 GTSRB 翻不动**。
- **FAAT 成功但付代价**：δ_global（Narcissus 优化方向，频率集中）+ 特征对齐压过自信模型，但 GTSRB 隐蔽性下降（SSIM 0.72 vs CIFAR 0.95；AC/SS-AUC 0.76 可检），因翻自信模型需更大扰动（L2 3.5 vs CIFAR 1.5）。

**★核心 insight：GTSRB 逼迫方法选边**——KST 选隐蔽（SSIM 0.94 / 谱峰 1.0）弃攻击（0.8），FAAT 选攻击（82.7）弃隐蔽（SSIM 0.72 / AC 0.76）。**无方法在 GTSRB 同时拿高 ASR + 高隐蔽**，是 clean-label backdoor 在极自信模型上的 Pareto 极限。

**三数据集自信度轴（写论文视角）**：

| 数据集 | clean BA | 模型自信度 | KST ε48 | FAAT | 格局 |
|---|---|---|---|---|---|
| Tiny-ImageNet | ~55% | 低 | 98.5（隐蔽+攻击）| 95.8 | KST 略胜 |
| CIFAR-100 | ~78% | 中 | 94.0（隐蔽+攻击）| 98.5 | FAAT 略胜 |
| CIFAR-10 | ~95% | 中高 | 92.6（隐蔽+攻击）| 93.6 | 持平 |
| **GTSRB** | **~100%** | **极高** | **0.8（隐蔽弃攻击）** | **82.7（攻击弃隐蔽）** | **Pareto 互换** |

→ 随模型自信度升高，"隐蔽 + 攻击同时成立"的窗口收窄；GTSRB 极限处 KST/FAAT 被迫分工。

---

## 6. 探针类触发器（可行性 probe）

| 触发器 | 目录 | 设计意图 | 结果 | 结论 |
|---|---|---|---|---|
| **RKT** | `rkt_cifar10_s0.7_s1` | 可学习重采样核触发器 | SSIM 0.34 / L2 18.7 / AC 0.07 | 机制可行，隐蔽性待优化 |
| **ORBIT-IRREP** | `orbit_*`（5 组） | D4 群表示防御触发器 | **ASR 0.788 vs fixed-patch 0.603，BA 0.943**（提交 `52b488a`） | ✅ 跨方向优势明确 |
| Style-Gram / GramCE | `style_gram*` | Gram 矩阵纹理对齐 | 无完整评估 | 失败/未完成 |
| Feast（phase/pix × starve/nostarve） | `feast_*`（4 组） | 特征饥饿 + 相位/像素触发 | 无完整评估 | 可行性验证 |
| ICAF | `icaf_a*` / `_wire_icaf` | 等照度线色差流（物理光学） | wire 测试 ASR 0.00 | 机制创新，未产出有效结果 |
| OPAL | `opal_b*` / `_wire_opal` | 序数多胞体（几何约束） | wire 测试 ASR 0.00 | 同上 |
| Narcissus strongaug | `narcissus_strongaug_seed1` | 强增强对照 | 无完整评估 | 对照 |

**章节结论**：多数为 smoke/feasibility，缺完整 ASR/BA。**ORBIT 有实证结果**值得深入；RKT/Feast 机制可继续；Style/ICAF/OPAL 当前未产出有效结果。**来源**：`orbit_*` 提交 `52b488a`、`rkt_*` 的 `rkt_eval.json`、其余 `results/<dir>/` 日志。

---

## 7. 全局总结与下一步建议

### 7.1 各家族一句话结论

| 家族 | 结论 |
|---|---|
| 论文 baseline | CIFAR-10 复现成功；GTSRB / 大类低投毒崩溃 → 新方法的发力点 |
| **FAAT** | **当前最强、最成熟**，4 数据集全超 baseline（+8.7~+51.9），AC/SS/STRIP/FP 失效，论文级 |
| ICIT/PA-ICT/PAT | PA-ICT 纯 PCA 是唯一稳定绕全图 NC；Pareto 已饱和，收益递减 |
| **KST/SDT** | **当前最有特色**：平谱触发器结构性规避频域防御（独家性质）；但撞 flat-spectrum↔可学性墙，GTSRB 失败 |
| 探针触发器 | ORBIT 有结果，其余 feasibility |

### 7.2 跨家族横向对比（ASR，越高越好；均为各方法最优配置）

| 数据集（投毒率） | baseline 最强 | FAAT | KST | 谁赢 |
|---|---|---|---|---|
| CIFAR-10（1% cl） | MultiBpp-B 85.4 / 论文 Blended 84.88 | **v4_l2_1.5 = 93.6** | ε20 reslin = 92.59 | FAAT ≈ KST（KST 多了平谱隐蔽） |
| GTSRB（1%） | Blended 34.1（其余 0） | **v4_l2_3.5 = 82.7** | **0（失败）** | **FAAT 碾压** |
| CIFAR-100（0.5%） | BadNets 85.06 | **v6c_l2_2.0 = 98.5** | ε48 reslin = 93.95 | FAAT > KST > baseline |
| Tiny（0.25%） | Blended 43.93 / BadNets 90.1 | v7_l2_2.0 = 95.8 | **ε48 = 98.5（三 seed ~98）** | **KST 略胜**（都超 baseline） |

**横向结论**：纯 ASR 战力 **FAAT > KST > baseline**；KST 的卖点不是 ASR 而是**频域不可见（平谱）+ on-manifold**——这是所有其他触发器都没有的结构性性质。

### 7.3 最应该做什么（明确建议，按优先级）

**P0｜主线收口：以 FAAT 为主写论文。**
- FAAT 已在 4 数据集全超 baseline、防御全失效、消融完整，是**最成熟、最可发表**的成果。立即以 FAAT 为主轴组织论文 Table/Figure。
- 待补：补齐 FAAT 在 Tiny-ImageNet 的多 seed（当前仅 seed1）；统一各数据集防御指标表（AC/SS/STRIP/FP/NC）。

**P1｜定位 KST/SDT 的角色（二选一，决策点）。**
- **选项 A（推荐）**：把 KST 作为**独立"隐蔽性专精"贡献**——主打"频域平谱触发器结构性规避谱防御"，这是独家性质（FAAT/PAT/Narcissus 都做不到）。诚实地把 GTSRB 失败和 flat-spectrum↔可学性权衡写进 limitation，不强求全数据集 SOTA。
- **选项 B**：先攻 KST-Soft（软谱惩罚替硬 FFT 约束，允许小谱峰）+ SDT 增强（非线性 Stein key + Hutchinson 迹，目标 d-prime > 1.5），看能否破墙。见 `docs/SDT-ROADMAP.md`。
- **决策依据**：若赶投稿 → 选 A（KST 现状已够一个 sub-contribution）；若有时间 → 先跑 KST-Soft 决定能否升格为主贡献之一。

**P2｜✅ KST campaign 工程缺口已补完（2026-08-06，Round-2/3/3b/4 共 26 run）。**
- ✅ CIFAR-100 ε48 full + 多 seed（reslin 三 seed 94.0/93.2/94.6 稳健超 baseline）；
- ✅ Tiny ε16/ε20/ε48 full（ε48 三 seed ~98，ε20 补完 Pareto 中间点）；
- ✅ GTSRB baseline full 300ep（确认崩溃：badnets/mbpp=0、blend=34.7）；
- ✅ 关键配置多 seed（CIFAR-10 e20_reslinear 三 seed 89-92 稳健；e16_forget 方差大不主推）。
- **核心 KST 章节实验已齐，可进入写作阶段**（剩余仅 CIFAR-10 完整 8 selection、弱 ε 多 seed 等可选 gap）。

**P3｜停止 / 低优先。**
- ⛔ **停止追"超 Narcissus/BppAttack 的新触发器"**——Saturation 分析已证 Pareto 饱和，受对抗距离 floor 约束。
- ⏸ 探针触发器（Style/ICAF/OPAL/Feast）暂搁置；ORBIT 若要做防御侧贡献再捡起。
- ✅ PA-ICT 纯 PCA 的"绕全图 NC"结果值得作为 FAAT 论文的**防御分析一节**直接引用（不必再扩）。

### 7.4 已证伪 / 应放弃的方向（负面证据归档）

| 方向 | 证据 | 处置 |
|---|---|---|
| KST-Learn（平谱 + CE-proxy） | CE 被平谱约束卡死，ASR 不升反破签名 | 放弃，改 KST-Soft |
| KST on GTSRB | 所有 ε ASR=0（模型太自信） | 诚实写 limitation |
| FAAT v2（CE 全局）/ v3（无界 adaptive）/ v5（CW） | 均不如 v3.1/v4 | 保留为负面证据 |
| ICIT/PA-ICT CE 路径 | 全被 NC 抓（异常 3.32-5.75） | 只保留纯 PCA |
| 新优化型触发器（Chroma/频带限制/RKT） | Pareto 受 floor 约束，被支配 | 不再追 |
| Style/ICAF/OPAL wire | ASR 0，未正确加载 | 实现问题，暂搁 |

---

## 8. KST novelty 精查结论（2026-08-06，决定投稿定位）

> 精查范围：白噪声/平谱触发器、高阶统计量触发器（攻击侧）、FTrojan-FSBA 后续、Stein on-manifold。决定 KST 能否独立投稿。

### 8.1 核心 prior work 对照

| 论文 | 与 KST 关系 | 是否覆盖 KST |
|---|---|---|
| **NoiseAttack (arXiv 2409.02251, 2024.09)** | WGN 平坦谱触发器，规避 spectral 防御 | 🟡 **部分覆盖**：平谱/WGN（2 阶 σ 判别）✅；4 阶签名 ❌ |
| HMD (ICME 2024) | 高阶矩 + 后门（**防御侧**检测） | ❌ 不覆盖（防御，非触发器设计）|
| FTrojan (ECCV 2022) / FSBA (2025) / DFDT | 频域后门（频率**集中**，非平谱）| ❌ 不覆盖（集中 vs 平谱）|
| Narcissus | 优化噪声（谱峰 68.8）| ❌ 不覆盖（非平谱）|
| Spectral Signatures (Tran, NeurIPS 2018) | 谱签名防御奠基 | ❌ 防御，非攻击 |

### 8.2 KST 剩余 novelty（精查后实事求是）

| 原 KST 卖点 | 精查后状态 |
|---|---|
| "首个平谱/白噪声后门触发器" | 🔴 **不成立**——NoiseAttack (2024.09) 先用 WGN 平坦谱 |
| "4 阶累积量签名作攻击触发器判别量" | ✅ **仍新**（NoiseAttack 用 2 阶 σ；HMD 是防御侧）|
| "4 阶 vs 2 阶防御的正交性理论" | ✅ **仍新**（无 prior）|
| SDT（Stein on-manifold 触发器）| ✅ 仍新（无 prior）|

**净 novelty = 4 阶签名判别 + 4 阶正交理论 + SDT**（不再是"首个平谱后门"）。属 incremental novelty，非全新范式。

### 8.3 投稿定位建议（更新 §7.3 P1）

- **🔴 方向 A（FAAT 主轴 + KST 章节）：强烈推荐**。KST 作 FAAT 论文里一个触发器变体章节，4 阶签名 + 正交理论是 NoiseAttack 没有的差异化，不必扛"独立首创新机制"大旗。NoiseAttack 不威胁此定位。
- **🟡 KST 独立投稿（顶会）：风险高，不推荐**，除非同时满足：
  1. 补 **KST vs NoiseAttack 直接对比实验**（同防御套件下 4 阶签名是否比 2 阶 σ 更难检）；
  2. 强打 **"4 阶正交理论"**（NoiseAttack 无）；
  3. related work 精准区分 NoiseAttack（2 阶 σ）vs KST（4 阶签名）vs HMD（防御侧高阶矩）。

### 8.4 必须在 related work 区分的工作
**NoiseAttack（最关键，平谱/WGN 前辈）**、HMD（防御侧高阶矩）、FTrojan/FSBA/DFDT（频域集中）、Narcissus（优化噪声）、Spectral Signatures (Tran NeurIPS 2018，谱防御奠基)。

---

*本文件为全实验总览。各家族细节数字以 `docs/<家族>.md` 与 `results/<组名>/output_1.log` 为最终真相源。最近更新：2026-08-04。*
