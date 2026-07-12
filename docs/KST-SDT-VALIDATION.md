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
