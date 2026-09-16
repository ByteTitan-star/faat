# Pilot Observations（P1 闸门材料）— 2026-09-12

> 数据：12 个 OOD-ABC checkpoint（`docs/pilot_metrics.csv` + `results_ood/*/stageB_metrics.json`）。
> 图：`docs/figs_pilot/fig_p1_p2.png`、`fig_p3.png`。脚本：`scripts/pilot_metrics.py`、`scripts/pilot_figures.py`。
> 定位：pilot evidence，非普遍机制确认（n=12，2 数据集，单架构）。

## Pilot Observation 1（继承自 48h 网格）：proxy 方向通用度 → victim ASR 单调（GTSRB）

| GTSRB | proxy_gen_nt（触发器阶段、victim 训练前可测） | victim ASR |
|---|---|---|
| cur | 0.871 | 84.5 |
| b | 0.681 | 65.8 |
| a | 0.577 / 0.575 | 49.5 / 50.7 |
| c | 0.001 | 3.3 / 4.0 |

完美单调（CIFAR-10 因天花板效应仅在低端破单调，作 sanity）。**含义：存在一个训练前可测、
无需 victim 训练的触发器侧统计量，在受限 regime 下预测 acquisition。**

## Pilot Observation 2（新，Fig P2）：acquisition = 触发特征"并入"目标锚点，而非"远离"

sep（||centroid(triggered nontarget) − c_target||）与通用度**负相关**：
c 臂 sep≈10.6–10.8（GTSRB，最大分离）但 ASR≈3.5；cur/b sep 2.5–4.7 且高 ASR。
pull（触发把非目标特征拉向锚点的量）c 臂最低（2.1 vs 5.9–7.2）。
**含义：后门获取的表征签名是"折叠进目标区域"，而"独立成簇"对应 non-acquisition——
低通用度触发器在表征空间反而最显眼。**

## Pilot Observation 3（新，Fig P3）：可学性与样本级检测性两个方向的解耦都已出现

- **学到了但检不出**（CIFAR-10 全体）：ASR 70–94% 而 AC 0.19–0.30 / SS 0.37–0.57（全部 ≤0.5，
  随机线以下）。
- **检出了但没学会**（GTSRB c 臂）：ASR≈3.5% 但 AC 0.61–0.62——检测器对"独立成簇"的投毒样本
  有响应，即使它们从未形成后门。
- GTSRB 内 ASR→AUC 非单调（a：ASR 50/AC 0.80 > cur：ASR 84.5/AC 0.69）：样本级检测响应
  跟随触发器的离群程度（sep），而非获取强度。
- **按 CLAUDE.md P1 闸门判据：GO**——"proxy generality → representation change → victim
  acquisition"序稳定；P3 呈 detector-dependent 非单调 = 结构性 decoupling 的初步证据。

## 指标口径备注（分层，见 CLAUDE.md）

AC/SS = training-sample 检测（poison-detection AUC）；STRIP/NC/FP 未入本 pilot，
STRIP 在 `faat/defenses.py`、FP 在 `stage_b_metrics` 同目录 `defenses.py:fine_pruning_defense`，
NC 未实现（后续按需）。所有响应均为 run 末 checkpoint 单次测量，无 seed 重复。

## 下一步（P2 入口，等作者确认）

1. **schedule 不变性检查**（CLAUDE.md 红线）：3 个代表条件 × {150ep, 300ep} 验证边界位置不漂。
2. 第一条边界曲线：以 **proxy_gen_nt 为可控轴**（calib_weight / pool 组成 / 触发器优化步数均可
   连续调节它），GTSRB 上 coarse-to-fine 扫描，纵轴 P_L（过渡带多 seed）。
3. 表征层扩展：神经元集中度、layer-wise CKA、增强一致性（零训练成本，现有 checkpoint 即可）。

## Pilot Observation 4（新，表征层扩展 `docs/pilot_representation.csv`）：acquisition 的通道集中度签名

触发引起的特征偏移在 512 维上的集中度（Gini / top-16 质量占比）随 acquisition 走高：

| GTSRB | gini(\|Δ\|) | conc_top16 | victim ASR |
|---|---|---|---|
| cur | 0.711 | 0.358 | 84.5 |
| a | 0.727 / 0.717 | 0.354 | 49.5 / 50.7 |
| b | 0.733 | 0.330 | 65.8 |
| **c（non-acq）** | **0.630 / 0.651** | **0.222 / 0.251** | 3.3 / 4.0 |

**学到的后门把偏移集中到少数通道；未获取的 c 臂偏移显著更弥散**——这是继 Obs1（行为前可测）
之后的第二个候选表征级标记，且同样在 victim 训练后的 checkpoint 上分离两个 regime。
CIFAR-10 各臂 gini 0.35–0.39 无分离（天花板效应一致）。

layer-wise 1−CKA（clean vs triggered）呈质变而非单调：cur 臂触发表征与 clean 几乎正交
（1−CKA=0.91）且获取最好；c 臂偏移反而较"温和"（0.30–0.37）但方向背离目标锚点——
结合 Obs2（c 的 sep≈11），非获取 regime 的几何是"自建新簇"，获取 regime 是"大位移并入目标区"。

增强一致性（两次随机增广的特征余弦）在所有 run 饱和于 0.98–0.99，无判别力；
需改为"触发偏移 Δ 的增广不变性"（cos(Δ(T₁x), Δ(T₂x))）再测——记为待改进，不作结论。

*以上均为 pilot evidence：n=12、单架构、run 末单次测量；跨 seed/架构/预算的稳定性待 P2。*

## Observation 5（Batch 2，2026-09-14）：相图成形——陡轴锐利、预算轴平坦

**预算轴平坦性跨触发器成立**（GTSRB，300ep，末20均 ASR，2 seed）：

| 投毒样本数 | 17 (0.05%) | 35 (0.1%) | 176 (0.5%) | 189 (1%) |
|---|---|---|---|---|
| a 臂（弱触发器） | 41.4 / 47.0 | 43.7 / 40.6 | 44.3 / 50.5 | 49.5 / 50.7 |
| **cur 臂（最强触发器）** | **80.9 / 75.5** | **82.7 / 79.0** | **84.5 / 82.4** | **84.5 / 82.8** |

弱触发器无预算阈值（17 样本即 40%+），**最强触发器同样无**（17 样本即 75%+，全程平坦）。
ED50 式预算相变在此 regime 的两个触发器家族上都不存在。

**校准轴（陡轴）过渡锐利**（arm c，ood_weight 插值）：

| w | 0（≡a 臂） | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| ASR | 49.5 / 50.7 | 5.8 / 2.7 | 3.7 / 9.8 | 3.3 / 4.0 |

四分之一强度的 OOD 校准就足以把 acquisition 完全压灭（3-10%），过渡位于 w∈(0, 0.25)
内、且形状接近跳变而非平滑剂量响应。**相图结论：可学性边界不在预算轴上，在触发器属性轴上，
且是锐利边界。**

**Schedule 不变性正式通过**：a@150ep = 46.8（300ep：49.5/50.7 区间内）；两数据集全部排序
保持 → 短 schedule screening 协议生效。

*警示：以上每点 2 seed、单架构、单数据集（GTSRB）；w∈(0,0.25) 的边界精确定位与 P_L(x)
形式化（过渡带 3+ seed）待下一批。*

## Observation 6（最终版，2026-09-16，n=51）：边界定位于 w*∈(0,0.05)，解耦完成量化

**校准轴阶跃（GTSRB，arm c，全 seed）**：w=0 → 10/10 获取（40.6–50.7）；w≥0.05 → 0/18 获取
（最高 11.5，3 seed × 6 个 w 值零例外）。P_L 从 1.00 直接跳到 0.00，**L50 < 0.05**（OOD 校准
强度不到 a 臂对比项 5% 即彻底关闭可学性）。

**预训练预测量（RQ③ 核心候选）**：Spearman(proxy_gen_nt, victim ASR) =
GTSRB **0.899**（n=38）/ CIFAR-10 0.800（n=9）/ CIFAR-100 1.000（n=4），全库 n=51。

**表征集中度分离**（GTSRB）：acquired gini 0.714±0.017（n=20）vs non-acq 0.633±0.026（n=15），
间距 ≈2.6σ 无重叠。

**检测解耦量化（RQ④）**：同一 AC 检测器对已获取后门（ASR≥20）的 AUC =
GTSRB 0.77±0.10 → CIFAR-10 0.25±0.05 → CIFAR-100 **0.02±0.01**（反相关）；
GTSRB 非获取触发器仍获 0.59 响应。检测有效性按 regime 从"中度"滑到"反向"，
且"检出"与"有害"双向不重合。

**跨 regime 校准严重度**：GTSRB（BA 99.9%）致命阶跃；CIFAR-100（BA 77%）−15 税（94→79）；
CIFAR-10（BA 95%）混合（−19/+2）。排序 cur≈b>a>c 三 regime 不变。

*遗留缺口（写作时如实声明）：单架构（ResNet18）；CIFAR-10/100 无非获取点（天花板）；
w 网格在 (0,0.05) 内未加密；检测层仅 AC/SS 样本级（STRIP/NC/FP 待补）。*
