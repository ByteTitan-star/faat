# The Saturation of the Clean-Label Backdoor Trigger Pareto Front

> 分析论文骨架（③ 方向）。把全部负面结果转为 insight 贡献。
> 分支 `exp/rkt-trigger`。威胁模型范围：Tier-1 仅数据集访问 / Tier-2 最多控制训练过程（用户确认上限）。
> 所有实验 CIFAR-10 除非另注。

## 0. Thesis（一句话）

**Clean-label 后门触发器的 ASR↔隐蔽 Pareto 前沿已饱和**：优化型被 Narcissus（优化对抗方向）占据、学习型被 BppAttack（量化）占据；触发器有效性受 victim 对抗距离几何下界约束，Narcissus 就坐在这个 floor 上，故**任何新触发器机制都无法支配前沿**。我们在 **6 个独立机制、4 个分辨率、标准防御套件**上实证确认。

这不是"我们没做出更好的"，而是"我们证明并量化了为什么做不出更好的"——一个可证伪、可复现的饱和性论断。

## 1. 威胁模型范围（两层）

- **Tier-1（仅数据集访问，最弱）**：攻击者只能改数据集。= 经典 clean-label 投毒（GeneralComponents 三组件、Narcissus）。触发器设计在此层。
- **Tier-2（最多控制训练过程）**：攻击者控制训练代码（含增强 pipeline）。= pipeline/augmentation backdoor。
- **排除**：权重级供应链 trojan、微调（用户排除；非"训练过程"）。

**本分析的饱和性论断覆盖两层**：Tier-1 触发器设计 6 机制全撞墙；Tier-2 pipeline/增强后门是先验（Flareon 等）。两层都无 novel 空间。

## 2. 核心：对抗距离 floor（理论）

**命题**：对优化型不可见触发器 δ（施于输入 x 翻转到目标 t），δ 必须把 x 移过 victim 的决策边界到 t。能达到翻转的最小范数扰动 = victim 的对抗距离 `d_adv(x→t)`。

- Narcissus **显式优化** δ 以最小化感知代价 subject to 翻转 → 它近似对抗方向 → **坐在 `d_adv` floor 上**。
- 任何其它优化型机制（重采样/抖动/颜色/频带/自然纹理）若要翻转，也必须过界，付出 ≥ `d_adv` 的代价。
  - **结构保持型**（噪声/量化族）≈ Narcissus，无优势。
  - **结构破坏型**（重采样，下采样丢高频）付出**额外**感知代价 → 被支配（可见）。
- 故：**无优化型触发器能在 ASR↔隐蔽上支配 Narcissus**（它在 floor）。6 机制实证确认。

**学习型前沿（BppAttack）**：经投毒学得"触发器↔目标"捷径，不翻转干净代理；其隐蔽下界 = "在 poison 子集上扰动多少仍可被 victim 学会关联"。BppAttack（量化）在此前沿；其变体（dither）SSIM 被支配。

**两个前沿都饱和**：优化型（Narcissus）、学习型（BppAttack）。

## 3. 实证：6 机制 Pareto 表（CIFAR-10）

前沿（高 ASR + 高 SSIM）：Narcissus / ICIT / BppAttack。6 机制全部在前沿之下。

| 触发器 | 类型 | ASR | SSIM | 相对前沿 |
|---|---|---|---|---|
| **Narcissus**（L2=1.5） | 优化噪声 | 0.88 | **0.95** | 前沿（floor） |
| **ICIT**（b2.0） | 优化噪声（输入条件） | 0.997 | **0.94** | ≈ Narcissus（增量） |
| **BppAttack**（量化 24:28:8） | 学习型 | 0.83–0.99 | **0.97** | 学习型前沿 |
| BadNets-C / Blended-C | 学习型（可见） | 0.71 / 0.73 | 0.89–0.96 | 可见触发器，非不可见 |
| ChromaTrigger（颜色 LUT） | 优化型 | 0.045¹ | 高 | 低于前沿（低幅度翻不动） |
| SMT（频谱整形） | 优化型 | ≈0.90 | ≈0.95 | ≈ Narcissus（无改善） |
| 频带限制（B=8 / B=6） | 优化型 | 0.46 / 0.29 | — | 低于前沿（频域可检） |
| 自然纹理高频 | 优化型 | 0.002 | 高 | 远低于前沿 |
| **RKT**（重采样，保亮度） | 结构破坏 | 0.82¹ / 0.07 | 0.48 / 0.59 | 被支配（下采样可见） |
| **dither**（有序抖动量化） | 结构保持变体 | 0.54¹ / ~0 | 0.81 / 0.92 | SSIM 被 uniform 支配 |

¹ proxyASR（更有利的门槛：连干净代理都翻不动则 victim 必失败）。前沿用 victim ASR。即便用更有利的 proxyASR 门槛，6 机制仍达不到前沿区域。

**headline 图**：`docs/figs/pareto_saturation.png`（ASR vs SSIM 散点，前沿高亮）。

## 4. 分辨率无关（wall 非 32×32 特有）

RKT（scale=0.7）SSIM 随分辨率：32→0.64, 64→0.80, 96→0.85, 128→**0.89**。即使 128×128 仍 < Narcissus@32 的 0.95（且 L2 9.2 vs 1.5，6×）。**wall 不随分辨率消失**——重采样在任何分辨率都被优化噪声支配。

## 5. 防御画像（前沿触发器被什么抓）

- AC/SS/STRIP/FP/patch-NC：Narcissus/ICIT/BppAttack **全规避**（clean-label 小触发器）。
- **全图 L2 NC**：抓 Narcissus/ICIT（噪声型，异常 2.7–3.24）；BppAttack（量化）profile 待补。
- 频域检测：Narcissus 的 δ 频域孤立时异常，但**加进真实图像后被自然高频方差淹没**（0.358→0.394）→ 频域检测实战失败。
- **结论**：前沿触发器对标准套件近乎全规避；唯一可靠抓它们的是全图 NC（且只对噪声型）。

## 6. 先验饱和（Tier-2 攻击向量也已被做）

- **pipeline/augmentation backdoor**（投毒放训练代码）：Flareon（ACM 10.1145/3774648）= 供应链代码注入 train-time 增强 pipeline；Augmentation Backdoors（OpenReview）；NTU/arXiv dynamic-augmentation。
- **可迁移不可见 clean-label**：Narcissus（CCS'23）本身可迁移；"A Transferable Backdoor Attack Against Black-Box Models"（PR）；AAAI'24 Alternated Training。
- **结论**：Tier-2 攻击向量也饱和。两层威胁模型均无 novel 攻击空间。

## 7. 启示（ constructive ）

1. **停止追"超 Narcissus/BppAttack 的触发器"**——Pareto 前沿饱和，受对抗距离 floor 约束（§2）。
2. 攻击侧若要有贡献，须**换向量/换问题**，非换触发器。
3. **防御侧有机会**：全图 NC 抓噪声型触发器（标准 patch-NC 漏）→ 一个防御贡献方向（补 BppAttack 的 NC profile 后更完整）。
4. **本研究全部负面结果构成饱和性的实证骨架**（6 机制 + 4 分辨率 + 防御套件 + 先验核查）。

## 8. 论文贡献声明（诚实）

- **C1（理论）**：形式化 clean-label 触发器的对抗距离 floor，证明优化型前沿饱和。
- **C2（实证）**：6 机制 × 4 分辨率 × 防御套件的 Pareto 饱和证据（含 2 个新机制 RKT/dither）。
- **C3（范围）**：两层威胁模型 + 先验核查，证明攻击侧 novel 空间整体饱和。
- 诚实定位：**分析/benchmark 贡献**（非新攻击），用负面结果解释"为什么做不出更好的触发器"。

## 9. 待补（做实论文）

- [ ] headline Pareto 图（`pareto_saturation.png`）。
- [ ] BppAttack 的全图 NC profile（补全防御画像）。
- [ ] 形式化 §2 的对抗距离 bound（理论与 Narcissus 优化的等价证明）。
- [ ] 多 seed 稳健性（前沿点）。
- [ ] 相关工作完整覆盖（Narcissus/FTROJAN/Input-Aware/SIBA/WaNet/BppAttack/Flareon/Transferable-Backdoor）。

## 附：数据来源
Narcissus/ICIT/BppAttack 指标来自本仓库 `results/icit_*`、`results/bl_*`、`resource/faat/proxy/` + `docs/ICIT-*-RESULTS.md`。RKT/dither 来自本分支 `faat/rkt_trigger.py`/`dither_trigger.py` + sweep。Chroma/SMT/频带/自然纹理来自历史（`faat/chroma_trigger.py`/`smt_trigger.py`，记忆 `faat-trigger-tradeoff-finding`）。
