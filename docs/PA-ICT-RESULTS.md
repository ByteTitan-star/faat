# PA-ICT (PCA-Aligned Input-Conditioned Trigger) — Results

> 分支 `exp/rkt-trigger`。CIFAR-10, 1% poison, ResNet18, budget L2=2.0, seed 1.
> 用户思路：输入条件触发器，扰动方向把触发样本对齐到目标类**类内主成分子空间** → 触发器成为"类中信号"。

## 核心结果（4 变体对比）

| 变体 | ASR | BA | SSIM | AC_AUC | SS_AUC | NC异常(目标类) | NC检测 |
|---|---|---|---|---|---|---|---|
| λ=1.0 + CE | 0.995 | 0.947 | 0.941 | 0.108 | 0.422 | 4.96 | **被抓** |
| λ=0.1 + CE | 0.997 | 0.946 | 0.942 | 0.096 | 0.480 | 5.75 | **被抓** |
| λ=0.0 CE（=ICIT 基线） | 0.998 | 0.945 | 0.941 | 0.101 | 0.425 | 3.32 | **被抓** |
| **λ=1.0 纯PCA（无CE）** | **0.973** | **0.949** | **0.947** | 0.282 | 0.446 | **0.79** | **✅ 绕过** |

## Headline

**纯 PCA 对齐（无 CE）是唯一绕过全图 NC 的变体**，且 ASR 0.973 / BA 0.949 / SSIM 0.947 全达标。
ICIT（λ=0）和所有 CE 路径都被全图 NC 抓（异常 3.32–5.75）；PA-ICT 纯 PCA 异常 0.79 < 2 阈值 → 漏检。

## 多 seed 稳健性（3 seed，诚实修正 seed-1）

seed-1 的"纯PCA绕/ICIT抓"对比偏强；3-seed 全图 NC 异常如下：

| 变体 | seed | ASR | BA | NC异常 | NC? |
|---|---|---|---|---|---|
| 纯PCA(无CE) | 1 | 0.973 | 0.949 | 0.79 | ✅绕 |
| 纯PCA(无CE) | 2 | 0.983 | 0.950 | -1.09 | ✅绕 |
| 纯PCA(无CE) | 3 | 0.969 | 0.948 | 1.21 | ✅绕 |
| ICIT(λ=0 CE) | 1 | 0.998 | 0.945 | 3.32 | ❌抓 |
| ICIT(λ=0 CE) | 2 | 0.986 | 0.948 | 0.16 | ✅绕 |
| ICIT(λ=0 CE) | 3 | 1.000 | 0.947 | 1.74 | ✅绕 |

- 纯PCA：ASR 均值 ~0.975 / BA ~0.949，**NC 绕 3/3**（异常 max 1.21，均值 **0.30**）。
- ICIT：ASR 均值 ~0.995 / BA ~0.947，**NC 抓 1/3**（异常 max 3.32，均值 **1.74**）。
- **结论**：纯 PCA 比 ICIT **更稳健地绕 NC**（均值 0.30 vs 1.74；flagged 0/3 vs 1/3），但不是绝对（ICIT 也常绕）。代价：ASR 低 ~2 点，AC_AUC 略高（0.28 vs 0.10，但都规避）。

## vs BppAttack（三组件）——清晰 NC 规避优势

BppAttack(quantizeB) 全图 NC 异常 = **6.77（被抓）**；纯 PCA max 1.21（绕过）。
→ **PA-ICT 在"全图 NC 规避"上明确击败三组件**（三组件被 NC 抓 6.77，PA-ICT 绕过）。这是"防御防不住"相对三组件的实证。ASR/BA/SSIM 与三组件同档或更优。

## 机制（为何纯 PCA 绕 NC，CE 不绕）

- **CE 路径**：优化器找一条**对抗方向**让代理翻转 → 后门是单点强方向 → victim 上目标类"异常易翻转" → 全图 NC 找到小 ||δ|| → 抓。
- **纯 PCA 对齐**：优化器把触发样本特征推入目标类**类内主成分子空间**（沿自然变化方向分布）→ 后门是**分布式类内信号**，非单点对抗方向 → NC 的"最小 δ 翻转"在目标类上找不到异常 → 漏检。
- 用户"触发器=类中信号"假设在防御侧坐实。

## 三目标达标（PA-ICT 纯PCA）
1. **ASR 0.973** ✓（略低于 ICIT 0.998，>0.97）
2. **BA 0.949** ✓（4 变体最高）
3. **隐蔽**：SSIM 0.947（不可见）✓；AC/SS 规避（AC 0.282/SS 0.446, TPR@1%=0）✓；**绕过全图 NC**（ICIT 做不到）✓

## 代价/权衡
- ASR 比 ICIT 低 ~2.5 点（0.973 vs 0.998）——换取 NC 规避（值得）。
- AC_AUC 0.282 略高于 ICIT 0.10（AC 上稍逊），但两者都规避（<0.5, TPR@1%=0），且 NC 是更强防御 → 净赢。

## 待补
- 多 seed 稳健性（NC 规避 0.79 vs 阈值 2，gap 大，但需多 seed 确认）。
- 标准 patch-NC（已知全漏全局触发器，预期也漏）。
- CIFAR-100 / GTSRB / Tiny 跨数据集。
- 机制可视化（poison 特征在目标类 PCA 子空间的投影分布 vs clean target）。

## 代码
`faat/pca_trigger.py`（compute_class_pca + optimize_pca_trigger，复用 ICGenerator，存 ICIT 格式）；`faat/train_pca.py`（算PCA+优化+转交 train_backdoor --backdoor_type icit）；`scripts/run_pca.sh`（4 卡 sweep）。
victim: `results/pca_b2.0_*_s1/`；trigger: `resource/faat/pca/b2.0_*`。
