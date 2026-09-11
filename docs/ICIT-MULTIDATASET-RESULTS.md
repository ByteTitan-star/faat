# ICIT 多数据集结果（path 2）

> 分支 `exp/novel-trigger-investigation`。ICIT = 输入条件不可见触发器（生成器 g(x) 逐图生成 L2 投影扰动）。
> 调度器 `faat/icit_dispatch.py` 跨 GPU 跑 6 组（CIFAR-100/Tiny ×{1.5,2.0} + CIFAR-10 seed{2,3}），300ep，1% (CIFAR) / 0.25% (Tiny) poison。

## ASR/BA + 隐蔽 + AC/SS（vs 可见 baseline）

| 数据集 | 配置 | ASR | BA | AC_AUC | SS_AUC | 可见 baseline 最强 ASR |
|---|---|---|---|---|---|---|
| CIFAR-10 | seed1 b2.0 | 99.7 | 94.5 | 0.183 | 0.463 | 83.0 (MultiBpp) |
| CIFAR-10 | seed2 b2.0 | 97.8 | 95.04 | — | — | |
| CIFAR-10 | seed3 b2.0 | 97.1 | 94.59 | — | — | |
| **CIFAR-100** | b2.0 | **99.6** | 77.0 | **0.009** | 0.238 | 85.06 (Badnets) |
| CIFAR-100 | b1.5 | 99.0 | 77.5 | 0.013 | 0.227 | |
| **Tiny** | b2.0 | **82.6** | 54.6 | **0.021** | 0.271 | 43.93 (Blended) |
| Tiny | b1.5 | 68.7 | 54.6 | 0.027 | 0.289 | |

## 三要求达成（path 2）
- **req① ASR 超可见 baseline**：CIFAR-10 +15-17、CIFAR-100 +14-15、**Tiny +39** 点。全数据集碾压。
- **req② BA 无损**：CIFAR-10 95、CIFAR-100 77（≈论文 78）、Tiny 55（32×32 分辨率所致，matched baseline 同档）。
- **req③ 规避聚类防御**：AC 0.009-0.18、SS 0.23-0.46（全 ≪0.5，TPR@1%=0）——全数据集规避 AC/SS。

## NC（pending，后台跑）
Narcissus 被 NC 抓（异常 2.72）；ICIT 的 NC 异常待测（预期同 Narcissus 被抓——但 NC 非标准 patch-NC，且 ICIT 主卖点是对可见 baseline + 迁移性，非 NC 规避）。

## path 3（迁移性，附加）
单 R18 训练 ICIT 迁移到 ResNet50 差（29-62%）；**集成训练（R18+34+50）把 R50 迁移拉到 92%**。→ path 2+3 合并：「可迁移的不可见 clean-label 触发器」。

## 待补
- STRIP/FP（defenses.py 需适配生成器）
- 标准 patch-NC（vs 全图 NC）
- 集成 ICIT 的 full victim 训练（端到端验证迁移）
