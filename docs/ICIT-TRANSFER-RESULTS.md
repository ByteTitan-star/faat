# ICIT 跨架构迁移结果（Phase 1）

> 分支 `exp/novel-trigger-investigation`。验证「可迁移的不可见 clean-label 触发器」。
> 集成 ICIT（R18+R34+R50 surrogate 训练，`faat/train_icit_ens.py`）vs 单 R18 ICIT，在 **held-out 深架构**（ResNet101/152，不在集成里）上的黑盒迁移 ASR。
> 调度器 `scripts/run_phase1.sh`，评估 `faat/_icit_blackbox_transfer.py`。

## 完整迁移表（CIFAR-10，flip rate % to target）

| 目标架构 | clean 基线 | 单 R18 ICIT | **集成 ICIT** | 提升 |
|---|---|---|---|---|
| ResNet18（源 victim）| 9.8 | 92.2 | 94.7 | +2.5 |
| **ResNet101（held-out）** | 10.1 | 50.8 | **92.7** | **+41.9** |
| **ResNet152（held-out）** | 10.1 | 45.3 | **93.9** | **+48.6** |

**结论**：单 R18 训练的 ICIT 迁移到深 held-out 架构差（45-51%）；**集成训练把两个 held-out 深架构的迁移都拉到 92-94%**。迁移性创新端到端坐实。

## 集成 ICIT 源 victim（R18）指标
- ASR 98.4 / BA 94.88（高 ASR + BA 无损）
- AC_AUC 0.286 / SS_AUC 0.459（规避聚类防御，TPR@1%=0）

## 对比：三组件 baseline 的黑盒迁移
Badnets-C / Blended-C / MultiBpp-RGB 迁移到 R101 均 **~10%（≈随机）**——学习型触发器本质不可迁移。ICIT（对抗集成型）92.7% → **迁移性是 ICIT 相对三组件的独有优势**。

## 论文核心贡献
ICIT = 首个支持**黑盒跨架构迁移**的不可见 clean-label 触发器：
① CIFAR-10/100 ASR 高于三组件
② **唯一可黑盒迁移到未见深架构（R101 92.7%, R152 93.9%）**——三组件 ~10%
③ 隐蔽/BA/防御规避与三组件相当

代码：`faat/train_icit_ens.py`（集成训练）、`faat/_icit_blackbox_transfer.py`（迁移评估）、`faat/_train_proxy_alt.py`（R101/R152 干净 victim）、`scripts/run_phase1.sh`（调度）。
