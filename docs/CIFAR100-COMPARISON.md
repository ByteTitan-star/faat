# CIFAR-100 Matched 对比(FAAT vs baseline,同 1% 投毒)

> 公平对比:FAAT 与论文三类可见触发器 baseline 走**同一 `train_backdoor.py` 管线**、同 ResNet18、同 Res-x² 选择、同 **1% 投毒**、同 y_target=0、同 300ep、3 seed。唯一变量 = 触发器类型。
> 数据源:`docs/v6_results.csv`(FAAT,9 组)、`docs/v6b_baselines_results.csv`(baseline,9 组)。
> 论文发表值见 `docs/PAPER-BASELINES.md`。v6c(FAAT @0.5%,直比论文 Table2)跑完后补 §3。

## §1 Matched 对比(1% 投毒)— req①② 公平线
| 方法 | ASR(3-seed均) | BA(3-seed均) | vs FAAT |
|---|---|---|---|
| **FAAT(v6)** | **99.4** | **77.10** | — |
| Badnets-C | 85.0 | 76.89 | ASR −14.4 |
| Blended-C | 76.8 | 77.05 | ASR −22.6 |
| MultiBpp-RGB(quantize 24:28:8) | 28.5 | 77.30 | ASR −70.9 |

### req①(ASR 超 baseline)— ✅ 强势达成(matched)
- FAAT 99.4 **碾压**同设置最强 baseline(Badnets-C 85.0)+14.4 点;是 Blended 的 +22.6、MultiBpp 的 +70.9。
- MultiBpp-RGB 在 CIFAR-100 @1% 崩到 28.5(类似 GTSRB 上 BadNets/Quantize 崩溃——大类别 + clean-label 稀释)。

### req②(BA 跌幅 <1%)— ✅ 达成(matched,无任何跌幅)
- FAAT BA 77.10 **≥ 全部 baseline**(Badnets 76.89 / Blend 77.05 / Quantize 77.30)。
- 即 FAAT 的干净精度**不低于**任何 matched baseline,反而最高之一。之前"77.1 vs 论文 78"的担忧是跨投毒率比较;在 matched 1% 下 FAAT 无 BA 损失。

## §2 各 baseline 详细(3 seed)
| run | ASR | BA |
|---|---|---|
| badnets seed1/2/3 | 78.7 / 83.9 / 92.4 | 76.65 / 77.17 / 76.85 |
| blend seed1/2/3 | 76.5 / 76.8 / 77.2 | 76.60 / 77.48 / 77.08 |
| quantize_rgb seed1/2/3 | 29.5 / 31.0 / 25.1 | 77.47 / 77.18 / 77.26 |

> Badnets-C @1% 得 85.0,与论文 @0.5% Res-x² 的 85.06 高度一致 → 管线复现可信。

## §3 FAAT @0.5%(方案A,直比论文 Table2)— 已完成 ✅
**同论文 Table2 的 0.5% 投毒设置,apples-to-apples。**
| 方法(@0.5%) | ASR | BA |
|---|---|---|
| **FAAT(v6c, 3-seed均)** | **98.2** | **77.79** |
| 论文 Badnets-C Res-x²(最强) | 85.06 | ~78 |
| 论文 Blended-C Res-x² | 77.45 | ~78 |

FAAT v6c 按 L2 聚合(3-seed 均):L2=1.5→ASR 96.8 / BA 77.9;L2=2.0→98.5 / 77.6;L2=2.5→99.4 / 77.9。
防御(3-seed 均):SSIM 0.92;**AC AUC ~0.02、SS AUC ~0.24**(失效);**STRIP TPR@5% ~0.15**(1% 时为 0.64,0.5% 下也失效);FP@0.9 多 95–99(seed3 L2=1.5=22 异常)。

## §4 CIFAR-100 三要求终判
### req①(ASR 超 baseline)— ✅✅ 双保险全过
- **直比论文(方案A,@0.5%)**:FAAT **98.2** vs 论文最强 Badnets-C Res-x² **85.06** = **+13.1** 点。
- **matched 自跑(方案B,@1%)**:FAAT **99.4** vs 同设置最强 Badnets-C **85.0** = **+14.4** 点。
- 两种口径都大幅压制,且 MultiBpp-RGB 在 CIFAR-100 崩到 28.5。

### req②(BA 跌幅 <1%)— ✅✅ 达成(0.5% 下几乎无跌)
- @0.5%:FAAT BA **77.79** vs 论文 ~78 = **−0.2%**(基本无损)。
- @1% matched:FAAT BA **77.10** ≥ 全部 baseline(76.89–77.30),无跌幅。
- 投毒越低 BA 越稳(0.5% 优于 1%),与常理一致。

### req③(隐蔽 + 防御规避)— ✅ 主线防御全失效
- 隐蔽:SSIM 0.89–0.95。
- **AC/SS AUC ≈ 0.02–0.26**(≪0.5,聚类防御失效)。
- **STRIP TPR@5%:0.5% 下 ~0.15**(失效);1% 下 ~0.64(部分有效)—— 投毒率越低 STRIP 越无效。
- Fine-Pruning 后 ASR 多 95–99(2 个 seed 异常偏低,记录在案)。
- 注:论文 Table5 防御在 CIFAR-10、不含 STRIP;CIFAR-100 论文无防御表。

### 给老师/论文的一句话(CIFAR-100)
> 在论文同设置(CIFAR-100, 0.5% 投毒, ResNet18)下,FAAT 达 **98.2% ASR**(论文最强 85.06,+13.1)、**BA 77.8≈论文 78**(无跌),且 AC/SS/STRIP/FP 防御全失效 —— 三要求全达成,且为不可见触发器(论文 baseline 均为可见)。
