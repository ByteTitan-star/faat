# Paper Outline — 方向 A（FAAT 主轴 + KST 章节）

> 起草于 2026-08-06。基于 `all_result.md` 全部实验 + novelty 精查结论（§8）。
> 目标：把现有实验串成一篇可写的论文，标出数据就绪度 + 写前待补。

---

## 1. 论文定位

- **主轴**：FAAT（Feature-Aligned Adaptive Trigger）—— poison-only clean-label backdoor。
- **三条贡献**：
  1. **FAAT 主方法**：4 数据集全超 baseline（+8.7~+51.9 ASR），AC/SS/STRIP/FP 防御失效；
  2. **KST 高阶签名触发器变体**：4 阶累积量签名 + 4 阶正交于 2 阶防御的理论（**NoiseAttack 没有的差异化**，作隐蔽性专精选项）；
  3. **极自信模型 stress test**（GTSRB）：揭示 ASR-vs-stealth Pareto 极限（三方分析 + 自信度轴）。
- **目标会议**：NeurIPS / ICLR（方法侧重）/ USENIX Security / S&P（安全侧重）——视投稿定位。

---

## 2. 章节结构 + 数据就绪度

| 章节 | 内容 | 核心数据来源 | 就绪度 |
|---|---|---|---|
| Abstract | 问题 + 方法 + 4 数据集 + 防御规避 | — | 待写 |
| 1. Introduction | clean-label 难 + 防御规避难 + 三条贡献 | — | 待写 |
| 2. Related Work | clean-label backdoor / 频域后门 / 高阶统计 — **必须区分 NoiseAttack / HMD / FTrojan / Narcissus** | `all_result.md` §8 | ✅ 素材齐 |
| 3. Method | 3.1 FAAT（δ_global + adaptive + L_align）/ 3.2 KST 变体（4 阶签名 + 平谱 + 正交理论）| `FAAT-EXPERIMENTS.md` / `KST-SDT-VALIDATION.md` | ✅ |
| 4. Setup | 4 数据集 / baseline 4 攻击 / 防御套件 / 指标 | `all_result.md` §1/§2 | ✅ |
| 5.1 Main results | FAAT vs baseline，4 数据集 | §3 + `bl_*` baseline | ✅ **Tiny 3 seed 已有**（v7 l2_2.0 均值 95.9）|
| 5.2 Defense evasion | AC/SS/STRIP/FP/NC 跨数据集 | `v4/v6/v7_results.csv`（全数据集有 AC/SS/STRIP/FP）| ✅ 已有 |
| 5.3 Ablation | 特征对齐 / adaptive / guidance scale | §3.6 | ✅ |
| 5.4 KST as stealth specialist | 4 阶签名 + 平谱 + 多 seed 稳健 | §5.9 | ✅ |
| 5.5 GTSRB stress test | 三方 Pareto + 自信度轴 | §5.9.5 | ✅ |
| 6. Limitations | GTSRB KST 失败 / 隐蔽性权衡 / 与 NoiseAttack 区分 | §5.9.5 / §8 | ✅ |
| 7. Conclusion | — | — | 待写 |

---

## 3. 写论文前待补实验

| 待补 | 章节 | 必要性 | 状态 |
|---|---|---|---|
| ~~FAAT Tiny 多 seed~~ | 5.1 主表 | ~~🔴 必须~~ | ✅ **已有**（v7 l2_{1.5,2.0,2.5} × seed{1,2,3} = 9 run；l2_2.0 均值 95.9，三 seed 稳健）|
| ~~FAAT 统一防御表~~ | 5.2 | ~~🟡 强烈建议~~ | ✅ **已有**（`v4/v6/v7_results.csv` 含 AC_AUC/SS_AUC/STRIP_TPR5/FP_ASR@0.9 全数据集）|
| KST vs NoiseAttack 直接对比 | 5.4（仅若 claim "4 阶优于 2 阶"）| 🟢 可选 | GPU 空时再说（当前 GPU 被他人占）|

**✅ 结论：论文数据 100% 就绪，可直接进入写作。GPU 被占也无所谓（纯写作不需 GPU）。**

---

## 4. 关键叙事线（写作时的主线）

1. **FAAT 主轴**：强 ASR + 4 数据集通用 + 防御规避 —— 论文主线。
2. **KST 隐蔽性专精**：4 阶签名 + 正交理论（NoiseAttack 没有的），作 FAAT 的隐蔽变体；不扛"首个平谱后门"旗（被 NoiseAttack 抢）。
3. **GTSRB stress test**：自信模型上三方 Pareto（baseline 崩 / KST 隐蔽弃攻击 / FAAT 攻击弃隐蔽），无人全赢 —— 深度 insight。
4. **自信度轴**：Tiny(BA55) → CIFAR-100(78) → CIFAR-10(95) → GTSRB(100)，方法表现随 victim 自信度变化 —— 比"4 数据集平铺"有深度。

---

## 5. Related Work 必须精准区分的 prior work

| prior work | 关系 | 论文里怎么写 |
|---|---|---|
| **NoiseAttack (2024.09)** | 平谱/WGN 触发器（2 阶 σ 判别）| "NoiseAttack 用 WGN 平坦谱以 σ 强度区分目标类；本文 KST 进一步用 **4 阶累积量签名**，提供 NoiseAttack 没有的高阶判别 + 4 阶正交理论" |
| **HMD (ICME 2024)** | 高阶矩（防御侧检测）| "HMD 用高阶矩检测后门模型；本文反过来用高阶签名**构造**触发器（攻击侧）" |
| **FTrojan/FSBA/DFDT** | 频域后门（频率集中）| "这些方法频率集中（易被谱防御检）；KST 强制平谱（δ 谱峰=1.0）" |
| **Narcissus** | 优化噪声（谱峰 68.8）| "Narcissus 频率集中；KST 同 ASR 下谱峰 1.0，Pareto 支配" |
| **Spectral Signatures (Tran 2018)** | 谱签名防御 | KST 规避它的理论依据（4 阶正交于 2 阶谱签名）|

---

## 6. 推荐写作顺序

1. 先写 §3 Method + §5 实验（数据齐，最实在）；
2. §2 Related Work（NoiseAttack 区分是关键，定调 novelty）；
3. §5.5 GTSRB stress test（最有 insight，亮点节）；
4. §1 Intro + Abstract（最后写，提炼贡献）；
5. §6 Limitations（诚实写 GTSRB + NoiseAttack 区分）。

---

## 7. 当前最大风险

**NoiseAttack 是 KST 的强相邻工作**。论文里必须：
- 不宣称"首个平谱后门"（被 NoiseAttack 抢）；
- 改打"4 阶签名 + 正交理论"差异化；
- 或把 KST 降为 FAAT 论文里的"隐蔽变体展示"（不必和 NoiseAttack 死磕）。
