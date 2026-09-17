# Design Implications（范式 #3 的设计委托书）— 节草稿 v1

> 表征论文的收尾节（Working title: *When Do Clean-Label Backdoors Become
> Learnable — and Detectable?*）。四条结论全部来自本仓库 51+9 run 相图与
> 机制链证据（docs/pilot_metrics.csv、pilot_representation.csv、
> pilot_representation_vgg16.csv、pilot_faat_family.csv）。每条都可直接
> 映射为下一代触发器范式（范式 #3）的设计约束。

## 1. Generality is the only lever（通用度是唯一杠杆）

proxy 通用度 g_nt 与 victim ASR 的映射跨越臂、数据集与触发器族成立
（GTSRB 全扫描点 ρ_s=0.89, n=38；含 FAAT 族跨数据集池化 ρ_s=0.69）。
FAAT 作为第二触发器族落点与 cur 臂同区（GTSRB g_nt 0.85 vs cur 0.87；
victim ASR 83.4 vs 84.5），而**自然方向全线不落区**（Scout A：目标类
top-1 主成分 Jacobian g_nt≈0.00，类均值模板 g_nt=0.30 @同预算）——
通用度不是"方向的自然属性"，必须**显式优化获得**。

→ 设计约束 #1：任何新触发器候选的第一道闸门是 trigger-stage proxy
generality（不训 victim，几分钟）；自然/启发式方向不值得上 victim。

## 2. Budget does not protect（预算不设防）

17 张投毒样本（0.05%）即可稳定获取（GTSRB a 臂 0.05%–1% 全平坦，
ASR 41–51；cur 臂同平坦 78–85）。投毒率轴在两个数量级内无 ED50 阈值。

→ 设计约束 #2：不要在投毒率上找安全边际——防御必须假设攻击者
只用极少量样本；检测器的样本预算假设要按 ≤0.05% 校准。

## 3. Folding into the target cluster is the structural blind spot
（折叠进目标簇 = 检测的结构性盲区）

获取成功的后门把触发特征并入目标类簇（pull>0，sep 低），AC 可分性
随之**反相**：victim ASR 越高、AC-AUC 越低（GTSRB 0.77 → CIFAR-10
0.25 → CIFAR-100 0.02）。同一机制对样本级 STRIP（TPR@5% ≤6%）与
模型级 NC（anomaly index <1）同样成立（docs/pilot_defenses_ext.csv）。
检出 ≠ 有害的反向解耦同样成立（GTSRB 非获取触发器仍获 AC 0.59）。

→ 设计约束 #3：追求"可学"与追求"不可检"在表征几何上是**同一个
动作**（向目标簇折叠）。范式 #3 若以提高通用度为目标，将自动加深
该盲区——防御侧需要与折叠正交的信号（时域/训练动力学/一致性问题）。

## 4. The overconfident victim is the only Pareto wall（极自信模型是唯一 Pareto 墙）

GTSRB（BA≈100%）是全部 51 run 中唯一"能学的就可检、隐蔽的就学不会"
的 regime：cur 臂（g_nt 0.87）付隐蔽代价（L2 3.5）；c 臂（隐蔽）完全
无法获取（0/18）。中低置信度 regime（CIFAR-10/100、Tiny、VGG16）无此
墙——触发器几乎完胜。

→ 设计约束 #4：范式 #3 的真正前沿不在"再多一个高 ASR 触发器"，
而在极自信 regime 打破 ASR–隐蔽 Pareto（GTSRB 是天然的试金石）。
预注册判据沿用：同 ASR 下更小 ‖δ‖。

---

*Scout A 记录（2026-09-17，预筛 NO-GO 已砍）*：目标类主成分方向三种构造
（PC-Jacobian / 类均值模板 / 类中心差 Jacobian）在 GTSRB proxy 上
g_nt ≤ 0.30（基准 0.87），按预注册纪律不上 victim。
*FAAT 族相图点*：docs/pilot_faat_family.csv（GTSRB 3 seed + C100 4 点）。
