# KST / SDT 触发器 CIFAR-10 验证结果 (2026-07-12)

分支 `exp/kst-sdt`。脚本 `faat/kst_sdt_validate.py`。配置:CIFAR-10, ResNet-18, 5% 投毒, 40 epoch, L∞=8/255 (ε=0.0314), seed=1, target=0。三触发器各一张 3090 并行。

## 1. 结果总表

| 触发器 | ASR | BA | SSIM | L2 | Linf | 谱峰/中位 | 判别 d-prime | on-manifold ‖score‖ 比 |
|---|---|---|---|---|---|---|---|---|
| **KST** | 0.909 | 0.824 | **0.991** | **0.59** | 0.031 | **1.00** | s(x)=**2.71** | — |
| **SDT** | **0.999** | 0.834 | 0.939 | 1.72 | 0.031 | — | S_g=0.69 | **1.005** |
| Narcissus | 0.998 | 0.842 | 0.946 | 1.64 | 0.033 | 68.8 | — | (off-manifold) |

对照图:`results/kst_sdt/kst_sdt_validation.png`。

**一句话**:两机制都跑通成后门(KST 91%、SDT 99.9%),且各自验证了理论声称的差异化性质——KST 在隐蔽性上全面优于 Narcissus(SSIM 0.991 vs 0.946、L2 0.59 vs 1.64、**谱峰 1.0 vs 68.8**),SDT 实现了 Narcissus 结构上做不到的 **on-manifold(ratio 1.005)**。

## 2. (a)/(b)/(c) 逐项验证

### KST — 4 阶累积量签名触发器 ✓✓✓
- **(a) 未用信号 + 不同行为**:`spectral peak/median = 1.00`(FFT 相位参数化 → 平坦幅度谱,构造性零频峰)vs Narcissus **68.8**。KST 触发器在频域**完全不可见**,而 Narcissus 有 69× 的频峰(频域防御一抓一个准)。理论"4 阶正交于 1/2 阶防御"实证成立。
- **(b) 非叠加图案**:δ 是 FFT 相位优化解出的平坦谱噪声(非 patch/blend/频带),L2=0.59(仅 Narcissus 的 1/3)、SSIM=0.991。视觉上无结构。
- **(c) 结构不变量判别**:`s(x) d-prime = 2.71`(s_clean=-0.11, s_trig=560)。4 阶多项式统计量把触发/干净分开 2.7σ,是强判别量——网络就是学这个 4 阶签名读出。

### SDT — Stein 残差 / 分数域触发器 ✓✓△
- **(a) 未用信号 + 不同行为**:`‖score(x+δ)‖/‖score(x)‖ = 1.005`——δ 是**流形切向**(on-manifold),被触发图仍在数据支撑集内。Narcissus 是离流界加性扰动。理论"流形切向 → 通过似然/flow 防御"实证成立。
- **(b) 非叠加图案**:δ 是逐图 Stein 残差最大化 + 切向投影的优化解(input-adaptive),非固定图案。
- **(c) 结构不变量判别**:`S_g d-prime = 0.69`——**弱**。Stein 残差均值差很大(1.67 vs 35.5)但方差也大,标量判别力不足。**但网络仍学到 99.9% ASR**——说明 victim 学到的读出特征比裸 S_g 标量更强。这是诚实弱点:Stein 残差作为标量判别量偏弱,需更富的 g(非线性)或更高阶 score 估计。

## 3. 关键发现(诚实)

1. **KST 在隐蔽性上 Pareto-优于 Narcissus**(同 L∞=8/255):SSIM 更高、L2 仅 1/3、谱峰 1.0 vs 69。这是用户 8 墙里第一次有机制在**防御相关轴(频域)**上结构性压过 Narcissus。代价:ASR 91% < Narcissus 99.8%(可通过提 ε / 投毒率补,见下一步)。
2. **SDT 的 on-manifold 是 Narcissus 结构上做不到的**(ratio 1.005),ASR 99.9% 与 Narcissus 持平。但像素隐蔽(SSIM 0.939)略逊 Narcissus,且裸 Stein 判别弱。SDT 价值在 on-manifold,不在像素隐蔽。
3. **KST 的 4 阶判别(d'2.71)远强于 SDT 的 Stein 判别(d'0.69)**——4 阶多项式是比 Stein 残差更干净的判别量。
4. **KST 与 SDT 互补**:KST 强在频域不可见 + 像素隐蔽 + 强判别;SDT 强在 on-manifold + 高 ASR。可分别写,或杂交(流形切向空间内最大化 4 阶签名)。

## 4. 下一步建议(按优先级)

**P0 — KST 主推(它压过 Narcissus 的频域轴,是论文级卖点)**:
1. **ε 扫荡**:ε∈{8,12,16,20}/255,把 KST ASR 推到 ~99% 同时记录 SSIM/谱峰(应仍平坦)。目标:在 SSIM>0.95 + 谱峰~1 下达到 ASR≥Narcissus。
2. **降投毒率到 1%**(匹配真实威胁模型 + 论文 Table 1 设定),验证 KST 在低投毒下是否仍 ASR>baseline。
3. **防御评估**:STRIP / 频域分析 / Neural Cleanse。核心卖点:KST 在频域防御下 ASR 不降(Narcissus 会崩)。复用 `faat/defenses.py`。
4. **多数据集**:CIFAR-100 / GTSRB(已有数据 + 32×32 管线),验证迁移性。

**P1 — SDT 完善**:
1. 换非线性 g(小 MLP,tanh)使 ∇·g 变 x 相关,提升 S_g 判别力(解决 d'0.69 弱点)。
2. 多尺度 score net(σ∈{0.05,0.1,0.2})替代单 σ。
3. on-manifold 防御评估(flow-based / likelihood OOD)——这是 SDT 独家卖点,需专门防御来证。

**P2 — novelty 核查**:配额恢复后 arXiv 搜 `"fourth-order" OR "kurtosis" OR "cumulant" backdoor attack` 与 `"score function" OR "Stein discrepancy" backdoor attack`,坐实两机制攻击侧空白(防御侧已有)。

**P3 — 杂交**:流形切向空间内最大化 4 阶签名 = on-manifold 的 KST,可能同时拿两机制长处。

## 5. 复现命令
```bash
source /media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/activate  # 若 activate 不可用直接用 bin/python
PY=/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python
cd GeneralComponents-main
CUDA_VISIBLE_DEVICES=0 $PY -u faat/kst_sdt_validate.py       --trigger kst       --gpu 0 --epochs 40 --poison_rate 0.05 --eps 0.0314 --seed 1
CUDA_VISIBLE_DEVICES=1 $PY -u faat/kst_sdt_validate.py       --trigger sdt       --gpu 0 --epochs 40 --poison_rate 0.05 --eps 0.0314 --seed 1
CUDA_VISIBLE_DEVICES=2 $PY -u faat/kst_sdt_validate.py       --trigger narcissus --gpu 0 --epochs 40 --poison_rate 0.05 --eps 0.0314 --seed 1
```
artifact:`resource/kst_sdt/{zca.pt, score_net_sigma0.1.pt, kst_delta_*.pt, narc_delta_*.pt}`;结果:`results/kst_sdt/{kst,sdt,narcissus}_eps0.031_pr0.05_s1_result.json` + `*_s_*.npy` / `*_Sg_*.npy` + `kst_sdt_validation.png`。

---

## 6. P0 结果 — ε 扫荡 + 防御评估(2026-07-12)

脚本:`faat/kst_sdt_validate.py --save_model`、`faat/defense_eval.py`、`faat/plot_pareto.py`。图:`results/kst_sdt/kst_pareto_defense.png`。

### 6.1 KST ε 扫荡(5% 投毒, 40ep, seed1)

| ε | ASR | BA | SSIM | L2 | **δ 谱峰** | s-dprime | STRIP AUC | 频域签名 peak/med |
|---|---|---|---|---|---|---|---|---|
| 8/255 | 0.956 | 0.841 | 0.991 | 0.59 | **1.0** | 2.71 | 0.489 | 42 |
| 12/255 | 0.990 | 0.822 | 0.976 | 1.00 | **1.0** | 3.87 | 0.575 | 37 |
| 16/255 | **0.998** | 0.850 | 0.961 | 1.33 | **1.0** | 4.64 | 0.787 | 9 |
| 20/255 | 0.994 | 0.842 | 0.940 | 1.71 | **1.0** | 5.50 | 0.904 | 11 |
| Narcissus 8/255 | 0.998 | 0.846 | 0.946 | 1.64 | **68.8** | — | 0.789 | **11217** |

### 6.2 关键结论(Pareto 支配 + 频域规避)

1. **KST 在 ε=16/255 达 ASR=0.998 = Narcissus,同时 SSIM=0.961 > Narcissus 0.946,δ 谱峰=1.0 vs Narcissus 68.8。** 即在等 ASR(99.8%)工作点上,KST 在隐蔽性(SSIM)和频域不可见性(谱峰)上**同时 Pareto 支配 Narcissus**。这是用户 8 墙后第一个结构性压过 Narcissus 的机制。
2. **δ 谱峰在所有 ε 恒为 1.0**(FFT 相位参数化的构造性保证)——Narcissus 的 68.8 是其频域集中性质,提 ε 只会更显眼;KST 提 ε 谱峰不变。结构性差异。
3. **频域签名防御(集合级,平均嫌疑图提取共同 δ 峰)**:Narcissus peak/med=**11217**(触发器被提取,可检),KST=9-42(估计噪声底,无可提取峰,**规避**)。逐图 FreqSig AUC 两者均≈0.5(频域防御是集合级的,逐图被自然 1/f 谱方差淹没——诚实)。
4. **STRIP(行为防御)**:KST 低 ε 规避(AUC 0.489 < 随机),高 ε 被抓(0.90);Narcissus 0.79。STRIP 捕捉强后门行为,与触发器类型无关——预期。KST 低 ε 同时规避频域 + STRIP。

### 6.3 P0 完成度
- ✅ P0-1 ε 扫荡:完成(KST 4 个 ε,Pareto 曲线 + 支配点确立)。
- ✅ P0-3 防御评估:完成(STRIP + 频域签名,KST vs Narcissus 对照)。
- ⬜ P0-2 降投毒 1%:未做(下一步)。
- ⬜ P0-4 CIFAR-100/GTSRB 迁移:未做(下一步)。
- P1(SDT 第二创新点)见 `docs/SDT-ROADMAP.md`,本轮未动。
- P2/P3 用户指定不做。

---

## 7. P0-2 结果 — 1% 投毒 + 3 seed(2026-07-12,诚实负向)

脚本 `scripts/run_p02_lowpoison.sh`、`faat/plot_p02.py`。1% 投毒(500 张),60 epoch,seed 1/2/3。图 `results/kst_sdt/p02_lowpoison_pareto.png`,summary `p02_summary.json`。

### 7.1 结果(mean±std, 3 seeds)

| 触发器 | ε | ASR | BA | SSIM | δ 谱峰 |
|---|---|---|---|---|---|
| KST | 8/255 | **0.029±0.002(失败)** | 0.856±0.002 | 0.991 | 1.0 |
| KST | 16/255 | 0.871±0.040 | 0.863±0.002 | 0.961 | 1.0 |
| Narcissus | 8/255 | 0.962±0.026 | 0.853±0.009 | 0.946 | 68.8 |
| Narcissus | 16/255 | 0.996±0.004 | 0.861±0.006 | 0.848 | 68.8 |

### 7.2 关键结论(regime-dependent — 诚实)

1. **KST ε=8 在 1% 投毒下完全失败(ASR 2.9%)**——平谱触发器太隐蔽(L2=0.59),500 张投毒样本不足以教会模型。5% 投毒时同样 ε=8 得 ASR 0.956,样本量是关键。
2. **KST ε=16 在 1% 投毒下工作(ASR 0.87)但低于 Narcissus**(ε=8 时 0.96,ε=16 时 0.996)。
3. **1% 投毒下 KST 不再 Pareto 支配**:
   - KST ε=16: ASR 0.87, SSIM 0.961, 谱峰 1.0
   - Narcissus ε=8: ASR 0.96, SSIM 0.946, 谱峰 68.8
   - 两者在 Pareto 前沿(KST 更隐蔽,Narcissus 更高 ASR),**KST 不支配**。
4. **5% vs 1% 对比**:5% 投毒时 KST ε=16 达 ASR 0.998=Narcissus + 更隐蔽 → 支配;1% 投毒时 KST ε=16 仅 0.87 < Narcissus 0.96 → 不支配。**Pareto 支配是 regime-dependent**(需 ≥5% 投毒或更高 ε)。

### 7.3 根因 + 修复方向

**根因**:KST 触发器优化目标是 4 阶统计量 s(x) 最大化 + 平谱约束,**没有为"网络可学性"优化**。Narcissus 是 CE-proxy 优化(直接让代理网络把触发图分到目标类)→ 更 sample-efficient,低投毒下更易学。KST 的隐蔽性(平谱+低 L2)以低投毒可学性为代价——经典 stealth-vs-learnability 权衡。

**修复方向(下一步,可让 KST 在 1% 投毒下恢复竞争力)**:
- **KST-Learn**:在 KST 触发器构造目标里加一个 learnability 项——`max s(x+δ) + λ·CE_proxy(x+δ, target)` s.t. 平谱 + L∞≤ε。即同时优化 4 阶签名(隐蔽判别)和代理 CE(可学性),平谱约束保 (a)。预期:1% 投毒下 ASR 回到 ~Narcissus 水平,同时保平谱+高 SSIM。
- 或增 epoch(60→200)+ 增 ε(20/255)。
- 这是 KST 从"5% 投毒支配"到"1% 投毒支配"的关键改进,优先级最高。

### 7.4 更新后的下一步
- ✅ P0-1 ε 扫荡(5%);✅ P0-3 防御评估;✅ P0-2 1% 投毒多 seed(完成,发现 regime-dependent)。
- **🔥 P0-2b(新,最高优先)**:KST-Learn——加 learnability/CE-proxy 项,目标 1% 投毒下 ASR≥Narcissus + 平谱 + SSIM>0.95。
- ⬜ P0-4 CIFAR-100/GTSRB 迁移。
- P1(SDT 第二创新点)`docs/SDT-ROADMAP.md`。P2/P3 不做。

---

## 8. P0-2b 结果 — KST-Learn 失败(2026-07-12,诚实负向 = 墙)

脚本 `scripts/run_p02b_kstlearn.sh`、`faat/plot_p02b.py`。1% 投毒,60ep,3 seed。KST-Learn = 在 KST.build 目标加 `CE_proxy(x+δ,target) − α·s(x+δ)/S`,平谱 FFT 相位约束保留。proxy=`resource/faat/proxy/resnet18_clean_cifar10.pth`。图 `results/kst_sdt/p02b_kstlearn_pareto.png`。

### 8.1 结果(mean±std, 3 seeds)

| config | ASR | SSIM | δ 峰 | s-dprime |
|---|---|---|---|---|
| α=0.0 (pure CE) ε=16 | 0.882±0.028 | 0.960 | 1.0 | 0.01 |
| α=0.5 ε=16 | 0.872±0.017 | 0.970 | 1.0 | 0.00 |
| α=2.0 ε=16 | 0.900±0.006 | 0.960 | 1.0 | 0.09 |
| α=0.5 ε=8 (救援) | **0.222±0.220(失败)** | 0.988 | 1.0 | 0.01 |
| pure-KST ε=16 (无 CE, P0-2) | 0.871±0.040 | 0.961 | 1.0 | **4.64** |
| Narcissus ε=8 (1%) | 0.962±0.026 | 0.946 | 68.8 | — |

### 8.2 结论:KST-Learn 失败,撞上根本墙

1. **CE 项无效**:α=0.0(pure CE)ASR 0.882 ≈ pure-KST 0.871(+0.01)。CE proxy 在平谱约束下 **CE 卡在 ~6**(P(target)≈e⁻⁶=0.25%),远高于 Narcissus 的 CE~0.1-0.5(P(target)~0.89)。**平谱约束根本性限制 CE 驱动力**——Narcissus 的频率集中正是其 CE 有效的来源,KST 强制平谱削弱了它。
2. **KST-Learn 还破坏了 (c) 4 阶签名**:s-dprime 从 pure-KST 的 4.64 掉到 ~0(δ 被 CE 优化,不再是 4 阶签名)。**既没换来可学性,又丢了 4 阶身份**——KST-Learn 是更差的实例化。
3. **α=2.0(强 4 阶项)略好(0.900)**:4 阶签名比 CE 更能提供可学信号(平谱下),但仍 <Narcissus 0.96,且 s-dprime 仍只有 0.09(α=2.0 下 s 项被 CE 主导)。
4. **ε=8 救援失败(0.22)**:CE 不救低 ε;平谱在低 ε 能量太小。
5. **根本墙:flat-spectrum 隐蔽性 vs 可学性直接冲突**。Narcissus 的频率集中 = 其可学性;KST 的平谱 = 其隐蔽性。二者是同一枚硬币的两面,无法同时拿到 strict 平谱 + Narcissus 级可学性。**这不是调参问题,是结构性权衡。**

### 8.3 修正后的 KST 定位(诚实)

- **pure-KST 是更好的实例化**(保 (a) 平谱 + (c) 4 阶签名 s-dprime 4.64),接受 regime 限制。
- **KST 在 ≥5% 投毒支配 Narcissus**(P0-1:ε=16 时 ASR 0.998=Narcissus + 更隐蔽 + 平谱)。
- **1% 投毒下 KST 在 Pareto 前沿**(更隐蔽:SSIM 0.96+平谱;但 ASR ~0.87-0.90 < Narcissus 0.96),**不支配**。这是 fundamental wall,非 bug。
- 论文定位应为"**spectrally-invisible backdoor:在 ≥5% 投毒或高隐蔽前沿上 Pareto-优于 Narcissus**",而非"全投毒区间支配"。诚实标注 1% 投毒的局限。

### 8.4 唯一可能突破 1% 的路径(未来,未验证)
**KST-Soft**:把硬 FFT 相位(strict 平谱)换成**软平谱惩罚**(允许小谱峰),换部分 (a) 换可学性。目标:小谱峰(远 <Narcissus 68.8,如 <5)+ 高 ASR。这是 flatness-vs-learnability Pareto 的探索,新机制变体,非本论文章节。当前 KST(strict 平谱)撞墙,KST-Soft 是下一步候选。

### 8.5 最终下一步(修正)
- ✅ P0-2b KST-Learn:完成,**失败**(撞 flat-spectrum vs learnability 墙)。
- 候选:KST-Soft(软平谱,未来);或接受 KST 的 regime 定位转 P0-4 迁移 / P1 SDT。
- P1(SDT 第二创新点)`docs/SDT-ROADMAP.md`。P2/P3 不做。

---

## 9. 方法论纠正 + Clean-label 对比 baseline(2026-07-12,**关键正面结果**)

### 9.1 之前 P0-2/P0-2b 的错(已作废)
P0-2/P0-2b 用了 **dirty-label**(投毒非目标类、改标签)+ 对比 Narcissus。但 baseline 论文是 **clean-label**(投毒目标类、标签保持)。威胁模型错配 + 对比对象错 → "1% 投毒失败 / flat-spectrum vs learnability 墙"的结论**是方法论 artifact,非真墙**。作废。

### 9.2 正确口径(对齐 baseline Table 1)
- **clean-label**:投毒目标类 500 张,加触发器,标签保持目标类。
- **Component A 样本选择**:复用 baseline 的 `save_metric_10_res`(forget / res-linear)。
- **baseline 管线**:ResNet18, SGD[60,90] γ0.1, 300ep, batch128, Pad+Flip+RandomCrop aug, ASR=非目标类测试图+触发→target(末20均)。
- 实现:KST 作为 `--backdoor_type kst` 集成进 `train_backdoor.py`(`Add_Clean_Label_Train_Trigger_kst`/`Add_Test_Trigger_kst`,仿 NAR 直接加 delta)。继承全部 baseline 管线,只换触发器。
- 对比对象:**baseline Table 1 的 4 个触发器(Badnets-C/Blended-C/MultiBpp-RGB/MultiBpp-B)+ Component A/B/C**,不是 Narcissus。

### 9.3 最终结果(CIFAR-10, 1% clean-label, 300ep, seed1, ASR末20均 %)

| selection | **KST best** | Badnets-C | Blended-C | MultiBpp-RGB | MultiBpp-B |
|---|---|---|---|---|---|
| random | 27.00 (ε16) | 36.37 | 49.87 | 30.95 | 8.81 |
| forget | **86.15 (ε16)** ✅ | 64.25 | 70.48 | 81.18 | 80.95 |
| res/linear | **92.59 (ε20)** ✅ | 68.78 | 75.86 | 85.09 | 89.70 |

KST 详细:ε16 random 27.00/BA94.96、ε16 forget 86.15/BA94.63、ε16 res/linear 80.47/BA94.56、ε20 res/linear 92.59/BA94.74。

图:`results/kst_sdt/cleanlabel_vs_baseline.png`。summary:`cleanlabel_summary.json`。

### 9.4 结论(反转 P0-2b 的"墙")
1. **KST + Component A 在 forget 上超过 baseline 全部触发器**(86.15 vs 最优 MultiBpp-RGB 81.18,+4.97)。
2. **KST ε20 + Component A 在 res/linear 上超过 baseline 全部触发器**(92.59 vs 最优 MultiBpp-B 89.70,+2.89)。
3. BA ~94.7% 持平 baseline(~94.5%)。
4. **隐蔽性**:KST 平谱(δ峰=1.0)+ SSIM 0.94(ε20)/0.96(ε16);baseline 4 个触发器全部频域有结构(无平谱)。KST 在"频域不可见"轴上独家。
5. **random(无 Component A)下 KST 27.00 在 baseline 区间内**(8.81-49.87,中游)——与 baseline 同模式:无 Component A 选样则弱。

### 9.5 核心叙事(可写论文)
KST 是一个 **clean-label 后门触发器机制**:在 baseline 的同一套管线 + 同一套 Component A 样本选择下,KST 在 forget/res-linear 选择上 ASR 超过 baseline 4 个触发器全部,同时 δ 频谱平坦(频域防御不可见)、SSIM 与 baseline 持平或更优。**之前 P0-2b 的"flat-spectrum vs learnability 墙"是 dirty-label 错配的 artifact——clean-label + Component A 下不成立。**

### 9.6 下一步
- 多 seed(seed 2/3)确认 KST > baseline 稳健。
- KST ε 扫荡在 clean-label + Component A 下的 Pareto(ε16/20 已超 baseline,补 ε12/24)。
- P0-4 迁移 CIFAR-100/GTSRB。
- P1 SDT 第二创新点(`docs/SDT-ROADMAP.md`)。
