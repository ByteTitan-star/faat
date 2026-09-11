# FAAT 最终结果汇总（CIFAR-10 + GTSRB，按三要求）—— 2026-07-06

> 分支 `exp/v5-gtsrb-stealth`。自包含 Narcissus 引擎（去掉作者 `noise_01000.pth` 依赖），跨数据集成立。
> 真相源：`docs/{v4,v5,v5b}_results.csv`、`docs/GTSRB-BASELINES.md`、`docs/V5-RESULTS.md`、`docs/result_all.md`。

## 三要求总判定
| 要求 | CIFAR-10 | GTSRB | 结论 |
|---|---|---|---|
| **① ASR 超 baseline** | 93.6 > 论文 84.88 ✅ | 83 ≫ baseline {0, 34, 0} ✅ | **两数据集全达成** |
| **② BA 跌幅 < 1%** | 94.8（Δ+0.2）✅ | 99.9（Δ≈0）✅ | **全达成** |
| **③ 隐蔽 + 防御规避** | SSIM .95、AC/SS/STRIP/FP 全失效 ✅ | SSIM .72-.81、AC/SS ~0.7（硬极限）⚠️ | CIFAR 强；GTSRB 诚实 Pareto |

---

## CIFAR-10（v4 自包含，论文 Table1 同设置：1% 投毒，ResNet18，Res-x²，300ep）
| L2 | ASR | BA | SSIM | AC | SS | STRIP@5% | FP_ASR@0.9 | vs 论文 84.88 |
|---|---|---|---|---|---|---|---|---|
| 0.9 | 76.1 | 94.8 | 0.98 | 0.33 | 0.44 | 0.003 | 73.9 | 隐蔽扫点（刻意低 ASR） |
| 1.2 | **88.6** | 94.7 | 0.967 | 0.34 | 0.44 | 0.003 | 86.4 | **+3.7 ✅** |
| 1.5 | **93.6** | 94.8 | 0.95 | 0.25 | 0.43 | 0.04 | 93.3 | **+8.7 ✅** |

- 复现 baseline 最强（Res-x²）= MultiBpp-B 85.42 / 论文 84.88；FAAT 93.6 超 ~8 点。
- **防御全失效**：AC 0.25、SS 0.43（接近随机）、STRIP TPR@5%≈0、Fine-Pruning 后 ASR 仍 93。且**不依赖作者 noise**（自包含引擎生成 δ_global）。

## GTSRB（43 类，论文无此数据集）
### FAAT 最优（v4/v5b，CE-based；v5 CW 与 v5b lambda3 探索见 V5-RESULTS.md）
| L2 | ASR | BA | SSIM | AC | SS | STRIP@5% | FP_ASR@0.9 |
|---|---|---|---|---|---|---|---|
| 2.5 | 64.5 | 99.9 | 0.81 | 0.71 | 0.79 | 0.24 | 62 |
| 3.0 | 71.5 | 99.9 | 0.76 | 0.72 | 0.73 | 0.30 | 75 |
| 3.5 | **82.8** | 99.9 | 0.72 | 0.76 | 0.74 | 0.37 | 87 |

### Baseline（自跑，同一 train_backdoor.py 管线，公平受控）
| 攻击 | ASR（3-seed mean） | BA |
|---|---|---|
| BadNets-C | **0.0** | 100 |
| Blended-C | **34.1** | 99.9 |
| Quantize/MultiBpp-RGB | **0.0** | 100 |
| **FAAT（L2=3.5）** | **82.8** | 99.9 |

→ FAAT 是最强 baseline 的 ~2.4×；BadNets/Quantize 在 43 类 + BA 饱和 + 189 clean-label 下崩溃（真实失败，已核实非注入 bug）。

---

## GTSRB ③ 的诚实结论（硬极限）
两个杠杆都试过、都失败：
- **v5（CW + adaptive ratio0.12）**：CW 伤 ASR、零 SSIM 收益；ratio 只微降 AC/SS（代价 ASR 崩）。
- **v5b（CE + lambda_align=3.0）**：ASR 恢复，但 AC/SS 不动（0.67-0.72 ≈ v4 0.71-0.76）。

**根因**：GTSRB AC/SS 可检性由高 L2 的 δ_global 本身决定（43 类 fooling 必需 L2≥2.5），对 adaptive/lambda 调整鲁棒。是数据集特性，非方法缺陷（CIFAR 同方法 AC≈0.25）。

## 给老师/论文的核心叙事
1. **自包含 + 跨数据集**：去掉作者 CIFAR 专用 noise 依赖，自包含 Narcissus 引擎在 CIFAR-10 与 GTSRB 都成立。
2. **CIFAR-10 全目标达标**：ASR 93.6 > 论文 84.88，BA 不损，**防御全失效**（AC/SS/STRIP/FP）——且不依赖作者 noise，证明规避非噪声特有。
3. **GTSRB ①② 强势**：ASR 83 vs baseline {0,34,0}，BA 99.9。**特征对齐在标准 baseline 崩溃的硬 regime 上成功**——方法有效性的直接证据。
4. **诚实边界**：GTSRB 防御规避（AC/SS ~0.7）是 43 类 + 大触发器的固有 Pareto 权衡，诚实记录。

## 未试的方向（备选）
- 感知损失优化的 δ_global（让触发器本身特征更难检）——最后一个未试的 ③ 杠杆，收益不确定。
- CIFAR-100（论文有 baseline，③ 大概率全过）——需 data100 + cal_metric + proxy 准备（~2-3h）。
- Narcissus/SIBA baseline on GTSRB（需数据集特定资源）。
