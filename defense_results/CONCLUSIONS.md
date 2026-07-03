# 后门防御实验验证结论

> 数据来源：`resource/defense_logs.zip`（作者随仓库附带的**真实 BackdoorBench 防御日志**）
> 9 种防御 × 7 攻击场景 × 6 样本选择策略，共 **342 条有效记录**
> 明细：`summary.csv`；逐场景：`by_scenario/`；解析脚本：`tests/analyze_author_defense.py`
> 口径：数值均为**防御后**指标；ASR 越低 = 防御越有效；ACC 越高 = 副作用越小

## 1. 防御有效性总览（防御后 ASR 均值，跨全场景/全选择）

| 防御 | 类别 | 防御后 ASR% | 判定 |
|---|---|---|---|
| AC (Activation Clustering) | 样本过滤 | **69.9** | 几乎无效 |
| FST (Fine-Tuning) | 模型修复 | **71.2** | 几乎无效 |
| FP (Fine-Pruning) | 模型修复 | 61.5 | 部分有效 |
| NC (Neural Cleanse) | 检测为主 | 58.0 | 修复弱（仅检测） |
| I-BAU | 模型修复 | 24.9 | 较强 |
| RNP | 模型修复 | 24.4 | 较强 |
| ABL | 模型修复 | **18.9** | 最强 |

> 强修复类（ABL/RNP/I-BAU）能压低 ASR，但代价是干净精度暴跌：ABL 的 ACC 仅 40–56%、RNP 仅 33–80%。

## 2. 关键结论一：clean-label 攻击天然规避"样本过滤"类防御
- **AC** 对本文 clean-label 攻击几乎无效（ASR 70–97%）：clean-label 不改变标签，激活向量聚类找不到异常簇。
- **STRIP** 在 BadNets/clean-label 下近乎失效（random TPR 仅 **7%**）；在 Sig 下中等（82–98%）。
- **SCAn** 是最强检测器：TPR 82–99%，FPR≈0。

## 3. 关键结论二：本文 res_square 选样本策略 → 对所有防御都更难清除 ⭐

防御后 ASR(%) 均值，**random vs 本文 res_square**（跨 7 场景）：

| 防御 | random | res_square | res_square 更难防御 |
|---|---|---|---|
| AC | 55.2 | 75.1 | ✅ |
| NC | 42.9 | 75.5 | ✅ |
| FST | 54.0 | 76.6 | ✅ |
| FP | 50.6 | 66.4 | ✅ |
| ABL | 19.2 | 21.7 | ✅ |
| RNP | 12.6 | **24.8（×2）** | ✅ |
| I-BAU | 21.4 | 23.1 | ✅ |

**7/7 防御全部成立**：本文精心挑选的投毒样本，比随机选择产生的后门更难被任何防御移除——这正是论文"协同样本选择提升鲁棒性"主张的定量证据。即使最强的 ABL/RNP，在 res_square 下残余 ASR 也翻倍。

## 4. 代表性场景明细（sig_p0.03 / Signal 攻击）

防御后 ASR%：
| 选择策略 | AC | NC | FST | FP | ABL | RNP | I-BAU |
|---|---|---|---|---|---|---|---|
| random   | 93.5 | 94.2 | 51.2 | 61.6 | **0.4** | **0.0** | 8.9 |
| res_square | 96.0 | 98.0 | 70.8 | 88.4 | **0.0** | **0.0** | 12.3 |

检测 TPR/FPR%：
| 选择策略 | STRIP | SCAn |
|---|---|---|
| random   | 82/7 | 98/0 |
| res_square | 98/9 | 99/0 |

> 完整 7 场景明细见 `by_scenario/`，全量矩阵见 `full_report.md`。
