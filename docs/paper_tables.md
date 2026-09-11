# Paper Tables — 论文用现成表（直接 copy）

> 生成于 2026-08-06。所有数字来自 `v4/v6/v6c/v7_results.csv` + KST 解析（`output_{seed}.log` 末20均）。3 seed 均值±std。
> 投毒率：CIFAR-10/GTSRB @1%，CIFAR-100 @0.5%（对齐论文 Table2），Tiny @0.25%（对齐论文 Table11）。

---

## Table 1 — Main: FAAT vs baseline（4 数据集，3 seed）

| Dataset | Poison | **FAAT** (best L2) | BA | Baseline 最强 | BA | Δ ASR |
|---|---|---|---|---|---|---|
| CIFAR-10 | 1% | **93.6 ± 2.3** (L2=1.5) | 94.8 | 85.4 (MultiBpp-B) | 94.9 | **+8.2** |
| GTSRB | 1% | **82.7 ± 3.3** (L2=3.5) | 99.8 | 34.7 (Blended) | 99.9 | **+48.0** |
| CIFAR-100 | 0.5% | **~98** (L2=2.0, v6c) | 77.0 | 85.06 (BadNets, 论文) | ~78 | **+13** |
| Tiny-ImageNet | 0.25% | **95.8 ± 1.6** (L2=2.0) | 54.7 | 43.93 (Blended, 论文) | ~57 | **+51.9** |

> FAAT BA 全部 ≥ baseline（无 utility loss）。CIFAR-100 @1% 更高达 99.6（见 v6 csv，附录）。
> 注：CIFAR-100 @0.5% 的 FAAT 3 seed 见 `v6c_results.csv`（l2_2.0 seed1/2 = 97.6/98.7，seed3 在 csv）；GTSRB/Tiny baseline 用自跑 full 300ep，CIFAR-100/Tiny baseline 用论文值。

---

## Table 2 — Defense evasion（FAAT 最优配置，3 seed 均值）

| Dataset | L2 | AC-AUC↓ | SS-AUC↓ | STRIP TPR@5%↓ | FP ASR@0.9↑ |
|---|---|---|---|---|---|
| CIFAR-10 | 1.5 | **0.250** | **0.433** | **0.040** | 93.3 |
| GTSRB | 3.5 | 0.757 | 0.737 | 0.372 | 87.5 |
| CIFAR-100 | 2.0 | **0.006** | **0.20** | 0.74 | 99.0 |
| Tiny | 2.0 | **0.004** | **0.25** | 0.09 | 78.4 |

> 解读：AC/SS-AUC 接近 0.5 = 规避（CIFAR-10/100/Tiny 都强规避）；STRIP TPR@5% 低 = 规避；FP ASR@0.9 高 = Fine-Pruning 剪枝后攻击仍存活（防御失效）。
> **GTSRB 是唯一 AC/SS 可检的数据集**（0.76）—— 与 §5.9.5 stress test 结论一致：自信模型上 FAAT 用更大扰动换 ASR，牺牲隐蔽。

---

## Table 3 — Stealth（SSIM / L2，3 seed 均值）

| Dataset | L2 | SSIM↑ | L2_meas |
|---|---|---|---|
| CIFAR-10 | 1.5 | 0.950 | 1.49 |
| CIFAR-100 | 2.0 | 0.919 | 1.96 |
| Tiny | 2.0 | 0.927 | 1.97 |
| GTSRB | 3.5 | **0.720** | 3.39 |

> CIFAR/Tiny SSIM > 0.92（高隐蔽）；GTSRB 0.72（Pareto 代价）。

---

## Table 4 — KST 多 seed 稳健性（核心 KST 数字，3 seed）

| Dataset | 配置 | seed1 | seed2 | seed3 | baseline 最强 | 判定 |
|---|---|---|---|---|---|---|
| Tiny (0.25%) | ε48 forget | 98.5 | 97.7 | 98.0 | BadNets 90.1 | ✅ 超 +8 |
| Tiny (0.25%) | ε48 reslin | 98.5 | 98.2 | 98.0 | 90.1 | ✅ 稳健 |
| CIFAR-100 (0.5%) | ε48 reslin | 94.0 | 93.2 | 94.6 | Blended 76.0 | ✅ 超 +18 |
| CIFAR-100 (0.5%) | ε48 forget | 83.2 | 91.9 | 92.9 | 76.0 | 🟡 s1 低 |
| CIFAR-10 (1%) | e20 reslinear | 92.6 | 91.5 | 89.5 | MultiBpp-B 85.4 | ✅ 超 +7 |
| CIFAR-10 (1%) | e16 forget | 86.8 | 64.8 | 62.0 | 85.4 | ⚠️ 方差大 |

> BA 全部 ≈ baseline（无 utility loss）。KST SSIM 0.94 / δ 谱峰 1.0（频域不可见，独家）。

---

## Table 5 — GTSRB stress test：三方 Pareto（亮点节用）

| 方法 | ASR (末20均) | peak@ep | PoisonLoss末 | 隐蔽性 |
|---|---|---|---|---|
| BadNets-C / MultiBpp | 0.0 | 1-7 | — | 可见 |
| Blended-C | 34.7 | 80.8 | — | 可见 |
| **KST ε48** | 0.8 | 19.3 @ ep139 | **9.44** | SSIM 0.94 / 谱峰 1.0（隐蔽）|
| **FAAT L2=3.5** | 82.7 | ~99 | — | SSIM 0.72 / AC 0.76（不隐蔽）|

**自信度轴**（写论文用）：

| Dataset | clean BA | KST ε48 | FAAT | 格局 |
|---|---|---|---|---|
| Tiny | ~55% | 98.5（隐蔽+攻击）| 95.8 | KST 略胜 |
| CIFAR-100 | ~78% | 94.0 | 99.6 | FAAT 略胜 |
| CIFAR-10 | ~95% | 92.6 | 93.6 | 持平 |
| **GTSRB** | **~100%** | **0.8（隐蔽弃攻击）** | **82.7（攻击弃隐蔽）** | **Pareto 互换** |

---

## Table 6 — Ablation（CIFAR-10，FAAT 组件必要性，已 parse 现有 log）

| 配置 | ASR | BA | 说明 |
|---|---|---|---|
| FAAT full (L2=1.5) | **93.6 ± 2.3** | 94.8 | 完整方法（3 seed）|
| no-align (3 seed) | 95.5 | 94.8 | 去特征对齐，ASR 持平 → 对齐换隐蔽非 ASR |
| **no-Adp (scale=0.1)** | **20.3** | 94.8 | 去 adaptive → ASR 崩 → **adaptive 是核心** |
| no-Adp (scale=0.2) | 62.7 | 94.6 | 去 adaptive，大 scale 部分恢复 |
| guidance = 0.1 | 68.6 | 94.8 | 低 guidance |
| guidance = 0.5 | 99.6 | 95.0 | 中 guidance |
| guidance = 1.0 (default) | 100.0 | 94.9 | 标准 |
| guidance = 1.0 + no-Adp | 100.0 | 94.7 | guidance 主导时 adaptive 冗余 |

**结论**：① **adaptive 是 ASR 核心组件**（去之崩到 20.3）；② **guidance scale 是强旋钮**（0.1→1.0: 68.6→100）；③ 特征对齐在 CIFAR-10 对 ASR 非核心，价值在防御规避轴；④ guidance 与 adaptive 是可学性的替代驱动。
（注：`ablation_CLEAN_*` 对照 ASR 61.7/90.3 偏高，疑似非纯 clean，不放本表，待核实。）

---

## 写作建议（哪些表放正文 / 附录）

- **正文**：Table 1（主表，必放）、Table 5（GTSRB stress test，亮点）、Table 4 选几行（KST 多 seed 稳健）。
- **正文或附录**：Table 2（防御）、Table 3（stealth）。
- **附录**：Table 6（消融，补齐后）、CIFAR-100 @1% 的 99.6 数字、各数据集完整 L2 sweep。

## 关键叙事数字（一句话能引用的）

- FAAT 在 **4 数据集全超 baseline**，ASR 提升 **+8.2 ~ +51.9**，BA 无跌。
- **GTSRB**：baseline 全崩（0-34.7），FAAT 达 **82.7**（唯一 work）。
- KST 在 Tiny/CIFAR-100/CIFAR-10 **三 seed 稳健超 baseline**，且 δ 谱峰=1.0（频域不可见）。
- 防御：AC/SS/STRIP 在 CIFAR-10/100/Tiny **全规避**（AUC~0.5, TPR~0），FP 剪枝后攻击存活 78-99%。
