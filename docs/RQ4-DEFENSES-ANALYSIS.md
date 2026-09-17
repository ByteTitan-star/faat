# RQ④ 检测解耦 — 四防御分层结论（2026-09-17 定稿）

> 数据：`docs/pilot_defenses_ext.csv`（18 个主网格 run × STRIP/NC/FP；AC/SS 见
> `docs/pilot_representation.csv` 管线与 HANDOFF §3）。评估协议与冻结仓库
> `faat/defenses.py` 一致（STRIP TPR@FPR=5%；NC anomaly <2 规避；FP90 = 剪除
> layer4.1.conv2 最不活跃 90% 通道）。

## 核心发现：AC 与 STRIP 的方向**相反**——"检测解耦"是防御特异、regime 特异的

| 防御 | GTSRB（BA≈100%） | CIFAR-10（BA≈93%） | CIFAR-100（BA≈77%） | 判词 |
|---|---|---|---|---|
| **AC**（聚类） | 0.77（抓强臂）→ c 臂 0.59 | 0.25 | **0.02** | **反相关**：越成功越瞎（折叠进目标簇） |
| **STRIP**（熵） | cur 0.88-0.89 / TPR5 0.77 | 0.39-0.73（全员 <0.5 除 cur） | a/b/cur **0.90-0.99 / TPR5 0.64-0.98** | **正相关**：越成功越可检（强触发器熵行为异常） |
| **NC**（反演） | 全体 0.81（max 0.88） | 0.43（max 0.99） | 0.62（max 0.77） | **结构性失效**：全部 <1，无一达 2 阈值 |
| **FP**（剪枝） | ASR 降幅 ≤5 | ≤8 | BA 崩至 43-47 但 ASR 仍 97-99 | **结构性失效**：信号分散于目标类主通道 |

## 三个写作级结论

### 1. 没有统一的"不可检"，只有"对哪个防御、在哪个 regime 不可检"
AC 的失效区（高 ASR 折叠后门）恰是 STRIP 的有效区（C100 a/b/cur 0.99/0.99/0.99；
GTSRB cur 0.89）；反之 AC 唯一保留信号的 GTSRB 非获取 c 臂（0.59）正是 STRIP
的盲区（0.42-0.48）。两防御的 ROC 在 regime 轴上**互补翻转**。

### 2. OOD 校准臂（c）在 CIFAR-100 上逃逸 STRIP——意外发现的防御效应
C100 上 a/b/cur 全被 STRIP 抓（0.90-0.99），唯独 **c 臂 AUC 0.337 / TPR5 0.05**
（仍获取 ASR 80.5）。OOD 负校准让触发样本的扰动-熵行为回归正常——这是 c 臂
触发器的一个**真实 STRIP 规避性质**，与它在 GTSRB 上的"无法获取"形成对照：
同一机制（OOD 校准）在不同 regime 一侧杀死可学性、一侧赋予 STRIP 规避。

### 3. NC/FP 对无 patch 全局触发器结构性失效——与族无关
NC 的"异常小反演触发器"假设对加性全局 δ + DCT 有界残差彻底不成立（18/18 <
1）；FP 剪最不活跃通道无法隔离嵌在目标类主通道的信号（C100 上 BA 崩 50 点后
ASR 仍 99）。诚实注记：C100 c 臂 FP90 ASR 80.5→7.2 系 BA 同崩至 43.5（模型
损毁，非后门清除）。

## 与 HANDOFF §3 原 RQ④ 表述的关系
原"AC 反相关 0.77→0.25→0.02"**保持成立**；本表新增：STRIP 正相关 +
c 臂 C100 STRIP 逃逸 + NC/FP 结构性失效全景。写作口径从"越成功越检不到"
修正为——**"learnability–detectability decoupling is defense-specific:
clustering-based detection anti-correlates with ASR, entropy-based detection
correlates with it, and OOD calibration flips the STRIP verdict within a
single dataset."**

## 方法注记
- GTSRB STRIP TPR5（a 臂 0.39-0.41）偏高源于极自信模型全部低熵的基线效应；
  AUC（0.42-0.65）是更稳的读数。
- 评估为最终 checkpoint 单点（与 KST/Narcissus 行同口径）；非末20均。
