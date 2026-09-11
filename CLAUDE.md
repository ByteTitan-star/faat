# CLAUDE.md — OOD-Calibrated Target-Specific Trigger（新论文主线）

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
