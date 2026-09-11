# 论文公开基线数据(GeneralComponents, arXiv 2509.19947 v2)

> **来源**:论文 *"A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers"* (NeurIPS 2025, arXiv:2509.19947v2, 2025-10-07)。作者代码 https://github.com/HITSZ-wzx/GeneralComponents.git
> **用途**:FAAT 三要求对比的**权威基准**(req① ASR 超 baseline / req② BA 跌幅<1% / req③ 防御规避)。
> **抓取**:2026-07-06 从 arXiv HTML v2 提取。下方表格按数据集重组(用户整理版),括号内为论文原表号。
> ⚠️ 投毒率(α)是公平性关键,见文末「FAAT 对比要点」。

---

## 1. CIFAR-10(论文 Table 1 / 4 / 5 / 3)

### 表1 — 样本选择对 ASR 的影响(投毒率 1%,ResNet18)[论文 Table 1]
| 选择方法 | Badnets-C ASR | Blended-C ASR | MultiBpp-B ASR | MultiBpp-RGB ASR |
|---|---|---|---|---|
| Random   | 37.24 | 53.41 | 1.37 | 1.16 |
| Loss     | 52.71 | 59.43 | 28.02 | 47.85 |
| Gradient | 52.56 | 58.45 | 38.26 | 53.28 |
| Forget   | 71.74 | 71.05 | 74.39 | 78.10 |
| Res-log  | **82.13** | 82.34 | 77.10 | 80.20 |
| Res-x    | 68.65 | 82.31 | 76.73 | **83.07** |
| Res-x²   | 78.76 | **84.88** | **82.54** | 83.88 |
| Res-eˣ   | 76.50 | 71.81 | 53.92 | 62.28 |

> BA 全部稳定 ~94–95%。**CIFAR-10 最强 baseline = Blended-C Res-x² 84.88**(FAAT v4 在 CIFAR-10 达 93.6,已 +8.7 ✅)。

### 表2 — 组件组合对 ASR 的影响(CIFAR-10,1% 与 2.5%)[论文 Table 4]
| 组合 | Badnets-C(1%) | Blended-C(1%) | Badnets-C(2.5%) | Blended-C(2.5%) |
|---|---|---|---|---|
| Vanilla  | 20.47 | 53.41 | 34.09 | 52.03 |
| +A       | 70.03 | 70.65 | 73.19 | 85.15 |
| +B       | 21.33 | 57.89 | 70.38 | 74.12 |
| +C       | 38.67 | 60.46 | 45.59 | 74.77 |
| +A+B     | 67.47 | 75.00 | 73.59 | 81.63 |
| **+A+C** | **86.15** | **84.13** | **89.85** | **94.32** |
| +B+C     | 58.57 | 70.66 | 70.75 | 85.20 |
| +A+B+C   | 77.67 | 77.51 | 84.49 | 87.54 |

### 表3 — 防御下的 ASR(CIFAR-10,投毒率 3%)[论文 Table 5]
| 防御 | Badnets-C(原) | Badnets-C(+A+C) | Blended-C(原) | Blended-C(+A+C) |
|---|---|---|---|---|
| ABL   | 18.8 | 1.6  | 57.6 | 1.8  |
| AC    | 14.4 | 68   | 52.4 | 85.5 |
| FP    | 8.0  | 51.2 | 39.8 | 92.8 |
| I-BAU | 18.8 | 14.8 | 28.3 | 22.9 |
| NC    | 10.5 | 81.2 | 57.1 | 96.1 |
| RNP   | 52.9 | 47.6 | 27.6 | 84.3 |
| FST   | 54.7 | 86.4 | 42.8 | 90.9 |

> 注:论文 Table 5 行=攻击方法、列=防御;此处转置便于阅读,数值一致。

### 表4 — MultiBpp(全局低强度)对比(CIFAR-10,2.5%)[论文 Table 3]
| 攻击 | ASR | BA |
|---|---|---|
| Benign(无攻击) | - | 95.0 |
| Base(32:32:32) | 8.2 | 94.8 |
| BppAttack(原版) | 12.5 | 94.5 |
| Blended-C | 66.4 | 94.3 |
| MultiBpp(255:255:8) | 68.6 | 94.8 |
| MultiBpp(24:48:8) | **76.6** | 94.7 |
| MultiBpp(8:255:255) | 84.1 | 94.7 |
| MultiBpp(255:8:255) | 72.2 | 94.3 |

---

## 2. CIFAR-100(论文 Table 2 / 9)

### 表5 — 样本选择对 ASR 的影响(投毒率 0.2% 与 0.5%)[论文 Table 2]
| 选择方法 | Badnets-C(0.2%) | Blended-C(0.2%) | Badnets-C(0.5%) | Blended-C(0.5%) |
|---|---|---|---|---|
| Random   | 7.49  | 40.48 | 51.49 | 65.55 |
| Loss     | 17.84 | 46.59 | 70.97 | 70.42 |
| Gradient | 25.25 | 53.03 | 82.02 | 72.51 |
| Forget   | 59.39 | 63.11 | 79.69 | 73.31 |
| Res-log  | 62.64 | 67.53 | 83.71 | 73.63 |
| **Res-x**| **80.48** | **73.48** | 84.05 | 76.36 |
| Res-x²   | 72.48 | 66.27 | **85.06** | **77.45** |
| Res-eˣ   | 52.14 | 60.04 | 71.41 | 72.20 |

> BA 全部稳定 ~78%。**CIFAR-100 最强 baseline:0.2%→Badnets-C Res-x 80.48 / Blended-C Res-x 73.48;0.5%→Badnets-C Res-x² 85.06 / Blended-C Res-x² 77.45**。
> ⚠️ **论文 CIFAR-100 用 0.2%/0.5% 投毒**(选目标类 20%/50%)。FAAT 当前 CIFAR-100 队列用 **1%**(选目标类 100%,即全部 500 样本)—— 投毒量是论文 max(0.5%)的 2×,**不是 apples-to-apples**。公平对比见文末。

### 表6 — Badnets-C 不同目标标签(CIFAR-100,0.2%)[论文 Table 9]
| 目标标签 | Forget ASR | Res-x ASR |
|---|---|---|
| 0  | 59.39 | **80.48** |
| 10 | 85.4  | **91.6** |
| 20 | 59.4  | **73.4** |
| 30 | 72.94 | **75.83** |
| 40 | 93.23 | **96.28** |
| 50 | 82.3  | **89.46** |
| 60 | 38.78 | **46.57** |
| 70 | **81.96** | 79.51 |
| 80 | 88.46 | **89.1** |

---

## 3. Tiny-ImageNet(论文 Table 11 & 12,合并)

### 表7 — 样本选择对 ASR 的影响(投毒率 0.25%,200 类,ResNet18)
| 选择方法 | Badnets-C ASR | Blended-C ASR |
|---|---|---|
| Vanilla(Random) | 17.06 | 27.71 |
| Loss     | 32.22 | 37.63 |
| Gradient | 31.74 | 38.74 |
| Forget   | 32.29 | 40.59 |
| Res-log  | 34.46 | 42.02 |
| Res-x    | 32.22 | 41.48 |
| **Res-x²** | **38.96** | **43.93** |
| Res-eˣ   | 32.29 | 38.31 |

> BA 全部稳定 ~57–58%。**Tiny-ImageNet 最强 baseline:Badnets-C Res-x² 38.96 / Blended-C Res-x² 43.93**(200 类、投毒率被限 <0.005,所以 ASR 普遍很低)。
> ⚠️ 论文用 **9×9 Badnets 触发器**(CIFAR 用 3×3)、64×64 图。FAAT 管线是 32×32 基础 —— 接入前要决定 resize 还是改管线。

---

## 4. SIG / CTRL 防御(CIFAR-10)[论文 Table 13]
| 防御 | SIG(原) | SIG(+ours) | CTRL(原) | CTRL(+ours) |
|---|---|---|---|---|
| ABL   | 0.4 | 0    | 6.5  | 6.6  |
| AC    | 93.5 | 97.5 | 84.2 | 96.5 |
| FP    | 61.6 | 88.0 | 94.9 | 99.2 |
| I-BAU | 8.9 | 42.4 | 39.3 | 65.7 |
| NC    | 94.2 | 98.0 | 91.3 | 94.8 |
| RNP   | 0.2 | 0    | 26.3 | 84.9 |
| FST   | 51.2 | 89.1 | 93.6 | 98.7 |

---

## FAAT 对比要点(写论文 / 判三要求用)

### 各数据集"最强 baseline"(req① 要超的线)
| 数据集 | 投毒率 | 最强 baseline ASR | 来源 |
|---|---|---|---|
| CIFAR-10 | 1% | **84.88**(Blended-C Res-x²) | 表1 |
| CIFAR-100 | 0.2% | **80.48**(Badnets-C Res-x) | 表5 |
| CIFAR-100 | 0.5% | **85.06**(Badnets-C Res-x²) | 表5 |
| Tiny-ImageNet | 0.25% | **43.93**(Blended-C Res-x²) | 表7 |

### req① 公平性 — 投毒率必须对齐
- 论文 CIFAR-10 / Tiny-ImageNet 的主表投毒率分别是 **1% / 0.25%**;FAAT 与之一致即可直接比。
- **CIFAR-100 是坑**:论文用 0.2%/0.5%,FAAT 当前队列用 1%(全目标类)。要公平,二选一:
  - (A) FAAT 也跑 0.2%/0.5%(生成新队列 `gen_queue_v6_cifar100` 改 poison_rate),与论文表5直接比;
  - (B) 自跑 Badnets-C/Blended-C/MultiBpp 在 **1%**(FAAT 同设置),做 matched 对比(同 GTSRB 做法)。
  - 推荐 **(B)**:与 GTSRB 一致、可复用 `train_backdoor.py`、且 1% 下 baseline 会更高(更难超,结论更强)。若 (B) 的 baseline 仍 ≪ FAAT 96-98%,则 req① 稳。
- FAAT 当前 CIFAR-100 中期预览 ASR **96-98%**(L2=1.5/2.0,1% 投毒)—— 即使按最严的 matched 1% baseline 口径,大概率仍压制。

### req② BA 基准
| 数据集 | 论文 BA(各方法均稳定) |
|---|---|
| CIFAR-10 | ~94–95% |
| CIFAR-100 | ~78% |
| Tiny-ImageNet | ~57–58% |
- FAAT CIFAR-100 中期 BA ~71.5%(CIFAR-100 ResNet18 合理区间,但**需对照 1% 干净基线**确认跌幅<1%;论文 0.2/0.5% 下 BA~78%)。

### req③ 防御(论文 Table 5,+A+C 是论文最强组合)
- 论文 +A+C 在 AC/FP/NC/RNP/FST 上 ASR 仍很高(68/51.2/81.2/47.6/86.4),ABL/I-BAU 能压住。
- FAAT 的目标:AC/SS/STRIP/FP 失效(CIFAR-10 v4 已验证)。CIFAR-100 防御跑完 `eval_all`(AC/SS/STRIP/FP)再判。
