# CLAUDE.md — 可学性边界 × OOD 校准 × 检测可观测性（防御表征研究主线）

> **核心命题（2026-09-12 定稿，作者核验后措辞）**：
> *This project does not optimize a new backdoor attack. It characterizes when clean-label
> backdoors become learnable under target-only poisoning, how OOD calibration shifts this
> acquisition boundary, and when learned backdoors become detectable—or remain
> observationally indistinguishable to existing defenses.*

## 🚫 Novelty 安全区（违反任何一条 = 表述错误，2026-09-12 作者核验定稿）

**不可 claim**（均有正式先例）：
- "首次研究 backdoor learnability"——Xian et al. ICML 2023 Adaptability Hypothesis（何时/为何学到后门的理论）
- "首次研究攻击强度与检测的关系"——Khaddaj et al. ICML 2023 *Rethinking Backdoor Attacks*：backdoor feature strength 定义 + 与 effectiveness 挂钩 + feature-strength 检测器 + AUROC
- "首次 target-only" / "首次 target-only + OOD"——Narcissus (CCS 2023) 就是 target-class + public OOD（OOD 参与 surrogate 获取/表征准备）
- "首次投毒率相变"——Zheng *Phase Transitions in Backdoor Learning*（ED50；注意：Harvard AI Safety 课程项目、LLM，只作 conceptually related / concurrent motivation，不作核心 methodological prior）
- "clean-label 可学性无理论"——Yu et al. ICML 2024 *Generalization Bound and New Algorithm for Clean-Label Backdoor Attack*

**安全的主张**：系统刻画 target-only clean-label 图像后门在多数据/训练约束下的 **empirical learnability boundary**；OOD 角色/距离/diversity 如何**移动**该边界；learnability–representation–detectability 三者**何时耦合、何时解耦**（decoupling region 是 RQ④ 的靶心，不是简单 ASR-AUROC 散点）。检测可行性的理论框架对照：Pichler et al. AISTATS 2024（检测作为假设检验的不可行性）——我们的故事："实证研究何种 learnability regime 下现有 detector 的假设成立/失效"。

Threat-model anchor = 自己上一篇 **Wicked Oddities (ICLR 2025)**：related work 逻辑 = "它证明 target-only 场景下 sample 异质性重要；我们问固定攻击族后，异质性 + 辅助 OOD 如何共同决定'是否被学到'与'学到后是否可检'"。

## 四个 RQ（2026-09-12 修订版）

1. **Learnability boundary**：target-only clean-label 投毒下，backdoor acquisition 如何随投毒预算/target 类支撑集/扰动可见性/增强/训练随机性迁移。产物 = **learnability phase map**：P_L(x)=P(acquisition|x)，报 L10/L50/L90 边界，不报单点 ASR。**seed 政策**：远离边界 2 seed，过渡带加密独立重复（相变区本质随机，2 seed 不足以定边界）。
2. **OOD calibration**：OOD 把边界往哪移（相对 no-OOD 与 ID non-target 参照）。OOD 是**实验变量**，不是攻击贡献。近域/远域/diversity 三轴。
3. **Representation mechanism**：哪些表征量**先于或预测** acquisition。Pilot Observation 1（已入档）：arm 级 proxy 方向通用度 → victim ASR 单调（见 `docs/ood_abc_results.md`，c 臂表述为 "crossed into the non-acquisition regime"，作低端 anchor 保留，不写 failed）。
4. **Detectability boundary**：**何时**可学性提升转化为可检测性提升，何时二者**解耦**。

## 防御指标必须分层（不可混为一张 AUROC 表）

| 方法 | 层级 | 报告什么 |
|---|---|---|
| Activation Clustering / Spectral Signatures | training-sample 检测 | poison-detection AUROC / TPR@FPR |
| STRIP | test-input 检测 | triggered-input detection AUROC |
| Neural Cleanse | model/class 级 | anomaly index（class 级检测） |
| Fine-Pruning | mitigation | ASR 降幅 vs BA 退化（不是检测 AUROC） |

## ⚠️ 实验设计红线

- **schedule 不变性是待验假设**：短 schedule（150ep）与 full 300ep 的边界位置/ordering 须先在少数代表条件上验证一致，才能把短 schedule 当 screening protocol；否则测的是 under-trained regime。
- P2 边界估计用 **coarse-to-fine**（围绕已观察到的过渡带加密），不做全矩形网格。
- **P1 = GO/NO-GO 闸门**：12 checkpoint pilot 若呈 "proxy generality → representation change → victim acquisition" 稳定序，即继续（detection 非单调反而更有意思 = 结构性 decoupling）。

> 创建：2026-09-11。基线导入自 `../GeneralComponents-main` @ `df72457`（tag `v5.0-paper-frozen`）。
> **旧仓库已冻结只读**（见其 `docs/BASELINE-FREEZE-2026-09-11.md`）。本仓库一切新产物写 `results_ood/` 与 `resource_ood/`，**绝不写回旧仓库或其 symlink 目标**。

## 这是什么项目

**科学问题**：Can arbitrary external OOD samples replace unavailable in-distribution non-target information for learning discriminative target-specific clean-label triggers?

**核心 novelty 主张**：OOD 不做 surrogate 预训练/数据增强，而是在 trigger learning 阶段充当**显式负校准参照**（negative calibration reference）——target samples 定义 trigger "应该是什么"，OOD samples 定义 trigger "不应该成为对任何输入都有效的 universal target direction"。

**当前阶段（48h 纪律）**：只验证机制是否活着，不加任何第四组件。

## ⚠️ 硬约束（违反任何一条 = 做错了）

1. **Novelty 证据 ≠ "仓库内没做过"**。仓库切割只证明与上一篇区分。论文级 novelty 必须单独文献核验，且 related work 必须精准区分：Narcissus（OOD→surrogate）、FFCBA/FMBA（需 ID out-of-class data）、BAAT（TDSC 2025, sample-specific attribute）、NoiseAttack（平谱 2 阶）、TriBA/FEAT/WPDA（频域）、AIBA（ViT attention）、2026 sample-customized feature alignment（需 source-label samples）。
2. **第一版方法只有 3 个组件**：target attraction + OOD negative calibration + 原有 perturbation/stealth 约束。不加：频域、attention、sample-specific generator、新 sample selection、改 poisoning pipeline、复杂 generative model。ASR 上去了也说不出哪部分贡献的。
3. **AC/SS 的定位**：P3 阶段、只作为**检测可分性分析**（representation-space 分析），不是方法目标。论文措辞用 "OOD calibration may reduce the separability of poisoned samples in representation space"，实验证实聚类分离度下降后才可下结论。不做 defense-evasion optimization。
4. **Go/No-Go（48h 末执行，不拖）**：
   - **GO**（hard regime，3 seed 稳定，BA 不明显掉）：同 perturbation budget 下 ASR 明显升；**或**相近 ASR 下允许更低 perturbation（"better learnability permits a weaker trigger"——这条叙事更值钱）。若换 OOD source 趋势保持 = 强 GO。
   - **NO-GO**（立即停，不堆模块）：只 CIFAR-10 有效；只对单个 OOD source 有效；换 architecture 增益消失；只有加大扰动才能超 A；**C ≈ B**（最危险——surrogate ≈ calibration，则核心 novelty 无实验证据）。
5. **数据集分工**：CIFAR-10 = sanity/compatibility only（其上 no-align 仍 96.4%，feature-shaping 对 ASR 非核心，**不得用 CIFAR-10 决定生死**）。**GTSRB = Go/No-Go 首选**（BA≈100%，现有方法需 L2 3.5 换 ASR 82.7——正是"learnability 是瓶颈"假设的检验场）；备选 CIFAR-100 @0.5%。

## A / B / C 严格定义（机制隔离，堵"你是不是只是多用了 OOD"）

| 组 | 定义 | OOD 的角色 |
|---|---|---|
| **A** | Positive-only target alignment（现有 FAAT/PA-ICT 逻辑，只用 target-class 内部信息） | 无 |
| **B** | CIFAR-10：B-Narcissus（作者 artifact `resource/faat/narcissus`，忠实设置）；其他数据集：B-OOD-Surrogate（OOD 只进 surrogate/pretraining，不进 specificity calibration） | 辅助表示资源 |
| **C** | OOD explicit calibration：target attraction + OOD 负参照参与 trigger 目标函数 | 显式校准参照 |

第一轮只回答 4 问：① C 在 hard regime 稳定优于 A？② C 优于 B？③ 匹配 poison ratio + 匹配 perturbation constraint 下增益仍在？④ 换 OOD source 仍在？

## 优先级

| 级 | 内容 | 备注 |
|---|---|---|
| P0 ✅ | 冻结旧仓库（tag `v5.0-paper-frozen`）+ 本仓库隔离创建 | 2026-09-11 完成 |
| P1 | 最小 OOD-calibration trigger 实现 + A/B/C × (CIFAR-10 sanity, GTSRB Go/No-Go) × 3 seed | 48h 内出判据 |
| P2 | cross-architecture（Proxy A→Victim B/C，尽早查 surrogate-specific artifact）；OOD source（near=CIFAR-100/far=SVHN 需下载）；**Oracle-ID calibration**（允许访问真实 ID non-target，仅作分析上界：期望排序 A < OOD-cal < Oracle-ID） |
| P3 | stealth/representation 分析、AC/SS 可分性、可视化 | 核心成立后才做 |

## 环境与 GPU

```bash
source /media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/activate
# python3.8.20 + torch1.11.0+cu113；300ep≈4-5h/run（3090）
```
- GPU 4×3090 **与他人共享**：只用空闲显存 ≥4GB 的卡；启动前 `nvidia-smi` 查看。
- 数据/selection metrics/预训练 proxy 都是**指向旧仓库的只读 symlink**（`data*`、`resource/save_metric_*`、`resource/faat/proxy`）。新 trigger artifact 写 `resource_ood/`，新训练日志写 `results_ood/<组名>/output_1.log`。

## 命名与记录纪律（沿用旧仓库规范）

```
results_ood/<attack>_<config>_<seed>/output_1.log   # 首行自动含完整 Namespace（train_backdoor.py 既有机制）
```
- 每跑完一组：`python parse_detail.py ./results_ood ./results_ood/_detail_summary.json --write-md docs/ood_results.md` + 手动更新进度。
- 新日志 tarball 备份到 `/media/hd0/wangxin/backup/`（日期后缀）+ GitHub 备份分支，约定见旧仓库 CLAUDE.md。
- 分支策略：`main` = 稳定版；实验开 `exp/xxx`；好→merge+tag，差→保留分支作负面证据。

## 已知坑（继承自旧仓库，勿重踩）

- `--selection grad`（不是 gradient）；`is_done` 判断用 grep 全局找 epoch-299；
- Blended-C 的 utils.py blend 补丁 [2,2,2]=0.2:0.2:0.2 已随代码带入本仓库；
- Res-log 与 Res-x² 在作者 stats pkl 上字节级相同；exp 分支 `get_stats` 已做下溢修补；
- GTSRB 上 KST 全失败、baseline 大多崩（极自信模型 regime）——这正是本项目的切入点，不是要复现的失败。
