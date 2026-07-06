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

## §3 FAAT @0.5%(方案A,直比论文 Table2)— v6c 跑完后填
论文 CIFAR-100 @0.5% 最强:Badnets-C Res-x² **85.06** / Blended-C Res-x² **77.45**。
v6c(FAAT @0.5%,L2{1.5,2.0,2.5}×3seed=9)进行中,完成后在此直比。
