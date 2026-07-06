# CIFAR-100 FAAT 结果(v6,1% 投毒)— 第一波 3/9 已完成

> 分支 `exp/v5-gtsrb-stealth`。自包含 Narcissus 引擎(v4 配方:CE+8000步+fix_global+adaptive_l2_max0.15)。
> 配置:CIFAR-100,100 类,ResNet18,Res-x² 选择,**1% 投毒**(全目标类 500 样本),y_target=0,300ep。
> 机器可读:`docs/v6_results.csv` / `docs/v6_results.json`。论文基准见 `docs/PAPER-BASELINES.md`。
> ⚠️ 1% 投毒 vs 论文 Table2 的 0.2%/0.5% —— 本表是"方法有效性"展示;req① 公平线待 v6b(matched 1% baseline)与 v6c(FAAT 0.5%)。

## 第一波已完成(末20 epoch 均 + stealth + 防御)
| run | ASR | BA | L2 | SSIM | AC_AUC | SS_AUC | STRIP@5%FPR | FP_ASR@0.9 |
|---|---|---|---|---|---|---|---|---|
| l2_1.5_seed1 | **98.9** | 77.2 | 1.47 | 0.949 | 0.003 | 0.184 | 0.666 | 96.5 |
| l2_1.5_seed2 | **99.2** | 76.9 | 1.48 | 0.949 | 0.008 | 0.206 | 0.760 | 95.2 |
| l2_2.0_seed1 | **99.6** | 77.2 | 1.96 | 0.919 | 0.006 | 0.205 | 0.742 | 99.6 |

ASR 三种子均 98.9–99.6,非常稳定。L2 越大 ASR 越高(1.5→98.9,2.0→99.6),SSIM 略降(0.949→0.919)仍隐蔽。

## 三要求初判(基于第一波 3 组)

### req① ASR 超 baseline —— 大概率 ✅(待 matched 确认)
- FAAT @1%:ASR **98.9–99.6**。
- 论文 CIFAR-100 最强 @0.5%:Badnets-C Res-x² **85.06**;@0.2%:Res-x **80.48**。
- 即使按最严的 matched 1% baseline(v6b 待跑),FAAT 99% 量级大概率压制。**待 v6b 跑完坐实。**

### req② BA 跌幅 <1% —— ⚠️ 边界,需干净基线确认
- FAAT BA **76.9–77.2**;论文 CIFAR-100 BA 稳定 ~**78**。
- 跌幅约 **0.8–1.1%**,卡在 req② 的 <1% 边界(seed2 的 76.9 略超)。
- **待确认**:v6b baseline(Badnets/Blended/Quantize @1%)的 BA 应 ~78;若 FAAT 77.2 vs baseline 78 → -0.8%(✅),但需补一个**干净无攻击** CIFAR-100 ResNet18 基线做绝对参照。
- 若边界紧张,缓解方向:L2=1.5 时 BA 更稳(77.2);或减小 adaptive 预算。

### req③ 隐蔽 + 防御规避 —— 大部分 ✅,STRIP 部分有效需注意
- **隐蔽**:SSIM 0.92–0.95(高=不可见),L2 1.47–1.96。✅
- **AC AUC ~0.003–0.008、SS AUC ~0.18–0.21**(均≪0.5)→ 聚类式防御基本失效。✅
- **Fine-Pruning(90% 剪枝)后 ASR 仍 95–99.6** → FP 失效。✅
- ⚠️ **STRIP TPR@5%FPR = 0.67–0.76** → STRIP 部分有效(能检 67–76% 投毒样本)。CIFAR-10 v4 时 STRIP≈0(全失效),CIFAR-100 上 STRIP 变强。这是 100 类的差异,需在论文里诚实讨论。
  - 注:论文 Table5(CIFAR-10)防御不含 STRIP;CIFAR-100 论文无防御表。所以 STRIP 不直接违反"作者有效防御我也要躲",但作为补充防御实验要记录。

## 后续(自主 chain 调度中, GPU1 全程排除)
1. **v6 剩余 6 组**(l2_2.0_seed2/2.5_seed1/2.5_seed2 + seed3 三组)— 跑完后补全本表。
2. **v6b: baselines @1%(方案B, 9 组)** — Badnets-C/Blended-C/MultiBpp-RGB → matched req① 公平线 + BA 参照。
3. **v6c: FAAT @0.5%(方案A, 9 组)** — 直接对标论文 Table2 的 85.06。
4. 全部完成后汇总 `docs/CIFAR100-COMPARISON.md` + req①②③ 终判。

chain 里程碑:`logs/cifar100_chain.log`。调度器状态:各 `logs/v6*/scheduler_state.json`。
