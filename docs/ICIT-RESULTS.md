# ICIT (Input-Conditioned Imperceptible Trigger) — 初步验证

> 分支 `exp/novel-trigger-investigation`。代码：`faat/ic_trigger.py`（生成器训练）+ `faat/apply_ic.py`（注入）+ `faat/train_icit.py`（runner）+ `train_backdoor.py --backdoor_type icit`。
> 核心主张：**首个规避 Neural Cleanse 的不可见 clean-label 触发器**（Narcissus 被 NC 抓住，ICIT 规避）。

## 背景（为什么 ICIT）
4 个触发器设计（Chroma/SMT/频带限制/自然纹理）撞墙 → 铁律：高 ASR 必依赖高频对抗内容=Narcissus 前沿。
Neural Cleanse 探测（`faat/_nc_probe.py`）发现 **NC 抓得住 Narcissus**（目标类异常指数 2.72>2），打破"饱和"。
ICIT 用**输入条件**触发器 g(x)（逐图生成不可见扰动；g 共享满足 C2，但 g(x) 随图变化）→ NC 的通用-δ 逆向抓不住 → 规避 NC。

## CIFAR-10 初步结果（seed1, 300ep, 1% poison, target=0）

| 触发器 | ASR | BA | SSIM | L2 | Neural Cleanse |
|---|---|---|---|---|---|
| Narcissus（对照，v4 l2_1.5）| ~96 | 94.6 | 0.954 | 1.5 | **被抓（异常 2.72）** |
| **ICIT budget=2.0** | **0.997** | **0.945** | **0.941** | 1.96 | **✅ 规避（无类>2）** |
| ICIT budget=1.5 | 0.98 | 0.946 | 0.967 | 1.48 | ❌ 被抓（目标类 flagged）|

**关键**：ICIT b2.0 同时达 高 ASR + 保 BA + 不可见 + **规避 NC**（而 NC 抓 Narcissus）。论文核心主张初步成立。

## 待办（严谨化）
1. **3 seed × {1.5, 2.0, 2.5}**：确认 b2.0 规避跨 seed 稳定；找 NC 规避的 budget 阈值；厘清为何 b1.5 被抓。
2. **适配 defenses.py 给 ICIT 生成器**：跑 AC/SS/STRIP/FP（应同 Narcissus 全规避）。
3. **标准 NC**（patch-based + L1 + TV，非简化 all-pixel L2）确认。
4. **Narcissus matched 对照**（同评估管线）。
5. **扩展 CIFAR-100 / Tiny-ImageNet**。

## 诚实标注
初步：1 seed、简化 NC（all-pixel L2）、b1.5 被抓原因未明、AC/SS/STRIP/FP 未跑。结论"ICIT 规避 NC"需上述 1-3 确认后才算坐实。
