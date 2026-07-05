# FAAT v5 GTSRB 结果与判定（2026-07-06）

> 分支 `exp/v5-gtsrb-stealth`。v5 目标：修 v4 唯一弱项（GTSRB ③ 隐蔽：AC/SS 0.66-0.80 可检、SSIM 0.72-0.81）。
> 两个改动：**CW-margin 全局触发器 + 2 万步**（ aimed at SSIM）+ **adaptive 预算等比缩放 ratio=0.12**（aimed at AC/SS）。

## v5 配置
- `global_mode=from_scratch, global_loss=cw, global_steps=20000, global_margin=10`
- `fix_global + adaptive_l2_ratio=0.12`（adaptive 预算 = 0.12×‖δ_global‖，相对 v4 写死 0.15）
- 其余同 v4：GTSRB, Res-x², 1% 投毒, 300 ep, seed{1,2,3}

## v5 完整结果（12/12 完成）
来源 `docs/v5_results.csv`。末 20 epoch 均值（3-seed）。

| L2 | ASR (3-seed mean) | BA | SSIM | AC_AUC | SS_AUC | STRIP@5% | FP_ASR@0.9 |
|---|---|---|---|---|---|---|---|
| 1.5 | **38.6** | 100.0 | **0.901** | **0.585** | 0.621 | 0.085 | 40.5 |
| 2.0 | 52.0 | 99.9 | 0.852 | 0.633 | 0.739 | 0.137 | 55.0 |
| 2.5 | 58.1 | 99.9 | 0.806 | 0.641 | 0.724 | 0.229 | 60.3 |
| 3.0 | **71.1** | 99.9 | 0.761 | 0.688 | 0.789 | 0.349 | 74.5 |

**L2=3.0 注**：v5 ASR 71.1 ≈ v4 71.5（持平），SSIM 0.761 与 v4 完全一致。即 CW 的 ASR 惩罚主要在低 L2（@2.5：58 vs 64.5），高 L2 时大预算补偿了 CW 低效；但 CW **依旧零 SSIM 收益** → 弃用结论不变。

## v5 vs v4 @ L2=2.5（同 L2 直接 A/B）
| 指标 | v5 (CW+ratio0.12) | v4 (CE+10k+abs0.15) | 判定 |
|---|---|---|---|
| ASR | 58.1 | **64.5** | v5 退 6.4 ❌ |
| BA | 99.9 | 99.9 | 持平 ✅ |
| SSIM | 0.806 | 0.805 | **几乎一样** —— CW 没改善 SSIM ❌ |
| AC_AUC | 0.641 | 0.712 | v5 好 0.07（微） |
| SS_AUC | 0.724 | 0.787 | v5 好 0.06（微） |

## 判定（决定 v5b 走向）
1. **CW-margin 失败**：proxyASR（CW 0.67 vs CE ~0.75 @L2=2.5）更低 → δ_global 更弱 → ASR 降；且 SSIM 同 L2 下与 v4 一致（SSIM 纯由 L2 决定，CW 无贡献）。**弃用 CW，回到 CE。**
2. **adaptive 等比缩放（ratio）只小幅改善 AC/SS**（0.71→0.64），代价是 ASR 崩（ratio0.12 @L2=2.5 给 adaptive 0.30 预算，过大 → 模型过度依赖 train-only adaptive，C2 共触发风险 → 测试 ASR 降）。**adaptive 饿死不是 AC/SS 的主因。**
3. **真正的 AC/SS 杠杆 = L_align 强度**（v4/v5 都用 lambda_align=1.0）。更强 L_align 把投毒特征更深地嵌进目标簇，不需放大扰动 → 不伤 ASR。

## v5b 设计（CE 救 ASR + L_align 压 AC/SS）
- `global_loss=ce, global_steps=10000`（回到 v4 的 CE，恢复 ASR）
- `adaptive_l2_ratio=0.09`（适度，介于 v4 的 0.06 等效与 v5 的 0.12 之间，避免共触发）
- **`lambda_align=3.0`**（3× 特征对齐 —— 单变量新杠杆）
- L2 {2.5, 3.0, 3.5} × seed{1,2,3} = 9 组，与 v4 同 L2 直接对比。

**v5b 成功标准**：ASR 恢复到 ≥ v4（64.5/71.5/82.8），同时 AC/SS 显著低于 v4（目标 <0.6）。

## GTSRB 的根本张力（诚实记录）
43 类 32×32 上高 ASR 需要大 L2（≥3.0），但大 L2 → AC/SS 可检、SSIM 低。这是 Pareto 前沿。
- v4 L2=3.5：ASR 82.8 / AC 0.76 / SSIM 0.72（高 ASR，可检）
- v5 L2=1.5：ASR 38.6 / AC 0.59 / SSIM 0.90（高隐蔽，低 ASR）
FAAT 的价值 = 把这条前沿往外推（v5b 的 lambda_align 即尝试）；**最终 req① 在 GTSRB 上靠 matched-stealth 对比撑**（论文无 GTSRB baseline）。
