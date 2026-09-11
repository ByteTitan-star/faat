# GTSRB Baseline 结果 + req① 判定（2026-07-06）

> 论文 **无 GTSRB 实验**，所以 req①（ASR 超 baseline）必须自跑 baseline 做对照。
> 全部走**同一管线** `train_backdoor.py`（ResNet18, Res-x² 选择, 1% clean-label 投毒, 300 ep, seed{1,2,3}），唯一变量是触发器类型 → 公平受控对比。
> 来源 `docs/gtsrb_baselines_results.csv`。

## Baseline ASR/BA（末 20 epoch 均）
| 攻击 | seed1 | seed2 | seed3 | mean ASR | BA |
|---|---|---|---|---|---|
| BadNets-C（3×3 固定 patch） | 0.0 | 0.0 | 0.0 | **0.0** | 100 |
| Blended-C（hello-kitty α=0.1） | 34.7 | 36.1 | 31.6 | **34.1** | 99.9 |
| Quantize / MultiBpp-RGB（24:28:8） | 0.0 | 0.0 | 0.0 | **0.0** | 100 |
| **FAAT（v4/v5b, L2=3.5）** | 86.2/82.8/83.3 | — | — | **~83** | 99.9 |

## 关键发现：标准 clean-label baseline 在 GTSRB 上崩溃
- **BadNets / Quantize：ASR=0（3 seed 完全一致）**。已核实非注入 bug（`Add_Test_Trigger_badnets` 正常应用 `checkboards[0]` patch；PoisonACC 全 300 epoch 恒 0，PoisonLoss≈10.8 不收敛）。
- **根因（真实，非 bug）**：GTSRB 43 类、模型 BA 几乎瞬间饱和 100%，仅 **189 个 clean-label 投毒样本**的**固定小触发器**（3×3 patch / 频带量化）被干净任务的强拟合稀释，学不出可泛化的后门。CIFAR（10 类、500 样本、更难饱和）下同样配置 BadNets ASR=70.6、Quantize=82.99 → 数据集难度差异所致。
- **Blended**：全图混合（扰动面积大）勉强学到 34%，但仍很弱。

## req① 在 GTSRB 的判定：✅ 强势达成（绝对压制，无需 matched-stealth 辩解）
| | FAAT | 最强 baseline |
|---|---|---|
| ASR | **64.5 / 71.5 / 83.3**（L2 2.5/3.0/3.5） | Blended 34.1（BadNets/Quantize=0） |
| BA | 99.9 | 99.9-100 |

**FAAT 在 GTSRB 上 ASR 是最强 baseline（Blended）的 ~2.4×，是 BadNets/Quantize 的 ∞×（它们为 0）。** 这正是 FAAT 核心论点的直接证据：**特征对齐自适应触发器在标准 clean-label baseline 崩溃的硬数据集 regime 上成功**。

## 给老师/论文的一句话
> 在 GTSRB（43 类交通标志）上，BadNets/Blended/Quantize 三个标准 clean-label baseline 在 1% 投毒下 ASR 分别为 0/34/0（模型干净精度饱和、固定触发器被稀释）；FAAT 凭特征对齐自适应触发器达到 83% ASR、99.9% BA —— 在 baseline 失效的 regime 上证明方法有效性。

## 后续（可选）
- baseline stealth（SSIM）：BadNets/Blended 可见（低 SSIM），Quantize 不可见。但因 baseline ASR 已≈0，stealth 对比意义不大。
- 可补 Narcissus/SIBA baseline（需数据集特定资源，当前未跑）。
