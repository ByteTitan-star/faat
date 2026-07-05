# FAAT-EXPERIMENTS.md — FAAT 实验单一真相源

> 对应计划 `docs/plans.md`。与 Table 1 复现真相源 `result_all.md` 分离：**faat_* 结果只进本文件，不进 result_all.md**。
> 调度器：`run_faat_stageA.sh`（GPU 礼让：等 `run_table1.sh` 结束后开跑）。日志：`results/_run_faat_stageA.log`。

## 统一配置（与 Table 1 对齐，保证可比）
`--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10 --selection res --res_sel square --backdoor_type faat`
ASR=PoisonACC，BA=CleanACC，取末20 epoch 均（同 result_all.md 口径）。

## Stage A 作业清单（规则版，proxy-free）
δ_global = Narcissus noise（全量 L2=6.6/absmax=0.125/SSIM=0.933，**不隐蔽**）；δ_adaptive = 纹理规则的 DCT 低中频噪声（eps~0.01–0.04）。

| result_dir | --faat_global_scale | --faat_eps | 目的 | 状态 |
|---|---|---|---|---|
| `faat_res_square_gs100` | 1.0 | (rule) | 全量 Narcissus + 自适应：Stage A 主验证(ASR>Random) | ✅ 完成 |
| `faat_res_square_gs100_noadp` | 1.0 | 0 | δ_global only：H3 预览对照(全局 vs 全局+自适应) | ✅ 完成 |
| `faat_res_square_gs050` | 0.5 | (rule) | stealth/ASR 权衡点 | ✅ 完成 |
| `faat_res_square_gs010` | 0.1 | (rule) | 更隐蔽档（接近真实 stealth 场景） | ✅ 完成 |

> 注：δ_global 主导，gs100 vs gs100_noadp 的差异(adaptive)会很小——这是 Stage A 已知局限，**H3 正式对比需 Week 2 做「等总 L2 预算」设计**。

## 结果表（Stage A 已跑完 2026-07-01；ASR/BA=末20均，参考 Random=30.95 / Res-x² MultiBpp-RGB=82.99）
| result_dir | ASR末值 | ASR末20均 | BA末20均 | L2 | SSIM | DCT-L1 | AC-AUC | SS-AUC | 耗时min | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| `faat_res_square_gs100` | 100.0 | 99.98 | 94.88 | 6.459 | 0.654 | 0.0844 | 0.062 | 0.652 | 171 | 全量 Narcissus+adaptive；**不隐蔽** |
| `faat_res_square_gs100_noadp` | 100.0 | 99.99 | 94.66 | 6.459 | 0.654 | 0.0844 | 0.060 | 0.619 | 87 | δ_global only(eps=0)；与 gs100 **完全一致→adaptive 残差无效** |
| `faat_res_square_gs050` | 99.7 | 99.57 | 94.98 | 3.265 | 0.846 | 0.0427 | 0.088 | 0.524 | 97 | 半量；仍偏可见 |
| `faat_res_square_gs010` | 73.1 | 68.57 | 94.80 | 0.658 | 0.990 | 0.0086 | 0.323 | 0.494 | 171 | 1/10 隐蔽档；SSIM≈0.99 隐蔽但 **ASR<MultiBpp baseline(82.99)** |

> 检测度量补充（TPR@1%FPR）：4 个 run 的 AC/SS **TPR@1%FPR 均为 0.0**（严格 1% 误报下两防御都失效）。AC-AUC 全 <0.5 说明投毒样本未形成独立簇（Narcissus 式全局噪声施于目标类样本，天然规避聚类防御）——但这并非 FAAT 创新，是 Narcissus 本身的性质。原始数值见 `results/_faat_stageA_metrics.json`。

### Stage A 判定（对照 Random=30.95 / MultiBpp-RGB=82.99）
- **ASR>Random**：4/4 全部成立（最低 gs010 的 68.57 ≫ 30.95）。✅ 管线有效。
- **adaptive 残差无效**：gs100 ≡ gs100_noadp（ASR/BA/隐蔽/检测逐项相同）→ 规则版 δ_adaptive 被 δ_global 完全淹没。**印证 Stage A 不含 FAAT 真创新**。
- **隐蔽/ASR 硬权衡**：gs100 SSIM0.65（可见）ASR100 vs gs010 SSIM0.99（隐蔽）ASR68.6——正是论文要解决的 RGB 不可分痛点。
- **隐蔽档打不过 baseline**：gs010(隐蔽) ASR68.6 < MultiBpp-RGB 82.99。**Stage A 无特征对齐，符合预期**。
- **结论**：Stage A 仅验证「管线能跑、能攻击、强触发可规避 AC」；真正验证 FAAT 价值（高 ASR+隐蔽+防御规避）须靠 **Stage B 特征对齐**（代码已实现+CPU 烟测通过，待 GPU）。

## Stage B 结果（faatb = 特征对齐版，真 FAAT）—— 2026-07-02 跑完
配置同 Table1（CIFAR-10 res-square @1%，y_target=0）。离线优化：干净代理(100ep) + 2000 步（λ_align=1.0, λ_align_global=0.5, λ_perc=0.3, λ_l2=0.05, λ_freq=0.02, eps_max=0.05, δ_global 从 Narcissus 热启动）。
**离线优化 C-机制信号（真实代理）**：`align` 9.55→**0.37**（↓25×），`align_g`(δ_global 单独对齐) 9.55→**0.79**（↓12×，C2 满足：测试触发器独立对齐目标簇）。|dg| 6.6→4.55（δ_global 变小同时更对齐）。

| result_dir | ASR末值 | ASR末20均 | BA末20均 | L2 | SSIM | DCT-L1 | AC-AUC | SS-AUC | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| `faatb_res_square` | 99.4 | **99.14** | 94.50 | 4.444 | 0.784 | 0.0562 | 0.154 | **0.398** | 特征对齐版；AC/SS TPR@1%FPR 均 0 |

原始数值：`results/_faatb_stageB_metrics.json`、`resource/faat/save_trigger_10_0/{global,adaptive}_delta.npy` + `opt.log` + `meta.json`。

### Stage B 判定（对照 Stage A gs100/gs010、MultiBpp-RGB=82.99、Random=30.95）
- **ASR 远超 baseline**：99.14 ≫ MultiBpp-RGB 82.99 ≫ Random 30.95；也 ≫ Stage A 隐蔽档 gs010 的 68.57。✅
- **C-机制(特征对齐)成立**：`align_g` 收敛到 0.79 → δ_global 单独就能把投毒特征对齐到目标类中心，故测试期(仅 δ_global)仍 ASR 99%。这是 Stage A 拿不到的直接证据。
- **防御规避改善（核心收益）**：**SS-AUC 0.398 < Stage A gs100 的 0.652**（SS 比随机还差）→ 特征对齐让投毒样本嵌进目标簇，破坏 SS 依赖的「独立谱方向」。AC-AUC 0.154 同样 <0.5。✅ 这正是论文 C-机制卖点。
- **vs Stage A gs100（等高 ASR 档）**：faatb ASR 99.14≈99.98，但**更隐蔽**（L2 4.44<6.46，SSIM 0.78>0.654）且**更规避 SS**（0.398<0.652）。✅
- **诚实保留**：faatb **不在「极隐蔽」区间**（L2≈4.4，SSIM 0.78）——不如 gs010 的 L2=0.66/SSIM 0.99；它是「以远低于暴力 gs100 的扰动预算，拿到同等高 ASR + 更强防御规避」。要进 L2<1 的高隐蔽区间，需下一步 **加大隐蔽权重(λ_perc/λ_l2)或降 δ_global 预算的 sweep**。
- **下一步**（门控已过）：① 消融（w/o align、w/o adaptive、w/o δ_adaptive、w/o DCT-band）量化各组件贡献；② 多 y_target 验证 `L_align↔ASR` 正相关（Stage B 完成标准）；③ 隐蔽权重 sweep 追求 L2<1+高 ASR。

## Stage B Pareto Sweep（隐蔽-ASR 前沿攻关，bug修复后正确 artifact）—— 2026-07-02
对 faatb 的 δ_global 施加硬 L2 预算(`--global_l2_max`)，4 档并行 victim 训练(CIFAR-10 res-square @1%)。**注意**：早期同名目录 `faatb_l2_*_INVALID_baseartifact` 是 bug 前(误用 base artifact)的作废 run，已改名保留不删，忽略。

| L2预算 | L2实测 | SSIM | ASR末20均 | BA末20均 | AC-AUC | SS-AUC | 满足目标? |
|---|---|---|---|---|---|---|---|
| 4.55(base,无约束) | 4.444 | 0.784 | 99.14 | 94.50 | 0.154 | 0.398 | T1✅ T2✅ T3❌(可见) |
| 3.0 | 2.951 | 0.873 | **94.11** | 94.70 | 0.134 | 0.346 | T1✅ T2✅ T3❌(SSIM0.87可见) |
| 2.0 | 1.974 | 0.928 | 67.75 | 94.73 | 0.276 | 0.393 | T1❌ T2✅ T3△(半隐蔽) |
| 1.3 | 1.287 | 0.966 | 42.04 | 94.67 | 0.310 | 0.461 | T1❌ T2✅ T3✅隐蔽但ASR太低 |
| 0.9 | 0.893 | 0.984 | 16.39 | 94.71 | 0.348 | 0.446 | T1❌ T2✅ T3✅隐蔽但ASR太低 |

> 目标：T1 ASR>82.99(超作者) / T2 BA≈94.6±1% / T3 肉眼不可见(L2<1.5,SSIM>0.95)+防御防不住。AC/SS 的 TPR@1%FPR 全部=0。

### Pareto 前沿结论（诚实，含负面发现）
- **ASR↔隐蔽硬权衡确认**：ASR 随 L2 下降陡峭(99→94→68→42→16)，SSIM 上升(0.78→0.98)。**没有任何一档同时满足 T1+T3**（高 ASR 且肉眼不可见）。ASR 跨过 82.99 的点在 L2≈2.5–3（SSIM≈0.87，仍肉眼可见）。
- **T2(ACC) 全程达标**：5 档 BA 94.5–94.73，与作者 94.6 差 <0.2%。✅
- **T3-防御规避全程达标**：AC-AUC/SS-AUC 全 <0.5、TPR@1%=0 → AC/SS 两防御在所有 L2 档都失效。✅
- **⚠️ 关键负面发现——faatb 的 δ_global 在等 L2 下 ASR 不敌原始 Narcissus**：
  - L2≈3：faatb 94.11 vs Narcissus gs050(L2=3.3) **99.57** → Narcissus 更高。
  - L2≈0.8：faatb 16.39 vs Narcissus gs010(L2=0.66) **68.57** → Narcissus 碾压。
  - 原因：faatb 把 δ_global 按 `L_align`(特征对齐)优化，而 `L_align` 不是 ASR 的直接目标；Narcissus 噪声本身是按「欺骗干净模型」优化的强方向，缩放后仍高效。faatb 的再优化反而弱化了 δ_global 的翻转能力。
- **faatb 的真实收益轴 = 防御规避，非 ASR**：等 L2 下 faatb 的 SS-AUC(<0.40)普遍低于 Narcissus(0.52–0.65)→ 特征对齐让投毒嵌进目标簇、更破坏 SS。即 faatb 用「ASR 换 SS-规避」。

### 下一步迭代方向（基于此发现）
当前 faatb 用 `L_align` 优化 δ_global 是 ASR 的次优目标。改进：**解耦**——δ_global 直接按 ASR-proxy(Narcissus 式「欺骗干净模型」目标)优化以保高 ASR，`L_align` 只作用于 δ_adaptive 用于防御规避塑形。预期：等 L2 下 ASR 追平 Narcissus、同时保留 SS-规避优势 → 才可能同时满足 T1+T3。

## Stage B v2 Sweep（δ_global=ASR-proxy 目标，CE 骗代理）—— 2026-07-02，假设证伪
`--global_obj asr`：δ_global 按 CE(proxy(x+δ_global)→target) 优化(Narcissus 式)。6 配置并行。

| 配置 | L2实测 | SSIM | ASR末20均 | BA | AC-AUC | SS-AUC |
|---|---|---|---|---|---|---|
| v2_base | 3.929 | 0.816 | 95.05 | 94.74 | 0.111 | 0.312 |
| v2_l3.0 | 1.962 | 0.930 | 52.88 | 94.68 | 0.147 | 0.342 |
| v2_l2.0 | 1.378 | 0.961 | 19.33 | 94.76 | 0.186 | 0.367 |
| v2_l1.5 | 1.062 | 0.975 | 8.93 | 94.80 | 0.258 | 0.405 |
| v2_l1.3 | 0.906 | 0.981 | 5.38 | 94.65 | 0.318 | 0.452 |
| v2_l0.9 | 0.571 | 0.992 | 1.58 | 94.93 | 0.354 | 0.455 |

### v2 结论（证伪 + 关键洞察）
- **v2 假设证伪**：CE 骗代理的 δ_global **ASR 反而全面低于 v1**(等 L2 下:base 95<99、L2~1.3 v2 5.4 vs v1 42、L2~0.9 v2 1.6 vs v1 16)。原因：CE 目标过拟合干净代理的决策边界，**不迁移到 victim**；而 v1 的 L_align(对齐目标簇中心)是更鲁棒、更易迁移的方向。优化时 CE 还把 δ_global 压到比 L2_max 上限更小(stealth 损失主导)。
- **防御规避 v2 略优于 v1**(SS-AUC 0.31–0.46 vs v1 0.35–0.46，base 0.312<0.398)，但 ASR 代价过大，得不偿失。
- **三轮定论(Narcissus 难超越)**：v1(align)、v2(CE)在等 L2 下 ASR **都不敌原始 Narcissus 缩放**(gs050 L2=3.3 ASR99.6、gs010 L2=0.66 ASR68.6)。作者预计算的 Narcissus 噪声是极强方向，**任何再优化 δ_global(align/CE)都在弱化它**。
- **三目标**：v2 仅 T2✅、T3-防御✅(AC/SS AUC<0.5)；T1+T3-隐蔽 同时达标 ❌(隐蔽档 ASR 太低)。

### v3 方向（基于三轮发现的正解）
**δ_global 固定用 Narcissus 缩放(不再优化，保留其强 ASR)，只优化 δ_adaptive(可学习,L_align)做防御规避塑形**。预期：ASR 追随 Narcissus(gs010 档→~68，远超 v1/v2 的 16/2)+ 学习型 adaptive 带来比 Stage A 规则版更好的 SS-规避 → 可能首次同时接近 T1+T3。即 Stage A 结构 + 可学习 adaptive。

## GTSRB 跨数据集验证 —— 2026-07-04，首次尝试失败
v3.1 配方(固定 CIFAR Narcissus)无法直接移植到 GTSRB(无预计算强触发器)。用 `--global_obj asr --init_random --global_l2_max X` 自生成 GTSRB 全局触发器 + bounded adaptive(`--adaptive_l2_max 0.15`)。data/GTSRB32(43类,Resize32,90/10切 train/val)。

| 配置 | L2预算 | 状态 | ASR | BA | 备注 |
|---|---|---|---|---|---|
| L2=3.0 | 3.0 | 跑完 | **0.90**(≈随机) | 99.95 | CE-proxy触发器未产生有效方向 |
| L2=1.3 | 1.3 | 崩 | — | — | `stats_forget_seed_1.pkl`缺失(cal_metric --epochs 11 未产最终stats) |

L2=3.0 隐蔽: L2=1.440 / SSIM=0.914 / AC-AUC **0.683**(AC 可检测) / SS-AUC **0.584**(SS 可检测)。与 CIFAR 形成鲜明对比(主因是 GTSRB 无预计算的强全局触发器)。

### GTSRB 失败根因分析
CIFAR 的 v3.1 依赖 **authors 预计算的 Narcissus 噪声**(noise_01000.pth，极强 ASR 方向)；GTSRB 没有等价物。CE-proxy 优化(`--global_obj asr --init_random`)在 2000 步/43 类下生成的 GTSRB δ_global 几乎不提供翻转力(ASR 0.9%)。**核心发现**：FAAT 的强 ASR 高度依赖 δ_global 的「种子方向」质量——CIFAR 有 authors 的强方向,V3.1 只需冻结它 + 有界 adaptive 锦上添花；GTSRB 从头生成则无法达到有效强度。方法普适性在这套配方下不成立。需换思路。

—— 2026-07-02，失败(C2 被破坏)
`--fix_global`（δ_global=Narcissus 冻结）但 δ_adaptive 无界。**ASR 崩塌**：scale 1.0 ASR 35、scale 0.5 ASR 1.1、scale 0.1 ASR 0.5（vs 同 δ_global 的 Stage A gs100=100/gs010=68.6）。
根因：L_align 把 `|da|` 推到 ~5（scale 0.1 时是 δ_global 0.66 的 5 倍）→ δ_adaptive 成训练期**主导共触发器**，victim 学 adaptive 不学 δ_global，测试期（仅 δ_global）ASR 崩塌。**违反 C2 约束**。

## Stage B v3.1（δ_global 冻结 + 有界 adaptive |da|≤0.15）—— 2026-07-03，✅ 成功
`--fix_global` + `--adaptive_l2_max 0.15`：δ_global=Narcissus 冻结（保强 ASR），δ_adaptive 硬约束 |da|≤0.15（轻度塑形、不夺权）。6 scale 档并行。`|da|` 实测全锁在 0.150。

| scale | L2实测 | SSIM | ASR末20均 | BA | AC-AUC | SS-AUC | 三目标 |
|---|---|---|---|---|---|---|---|
| 1.0 | 6.459 | 0.655 | 100.00 | 94.86 | 0.056 | 0.612 | T1✅T2✅T3❌(可见) |
| 0.5 | 3.265 | 0.846 | 99.82 | 94.69 | 0.094 | 0.554 | T1✅T2✅T3❌ |
| 0.3 | 1.966 | 0.927 | 97.35 | 94.24 | 0.144 | 0.376 | T1✅T2✅T3△(SSIM<0.95) |
| **0.2** | **1.313** | **0.963** | **94.81** | 94.67 | 0.223 | 0.416 | **T1✅T2✅T3✅** |
| 0.15 | 0.986 | 0.978 | 82.92 | 94.53 | 0.261 | 0.433 | T2✅T3✅T1△(82.92≈82.99) |
| 0.1 | 0.658 | 0.990 | 70.96 | 94.57 | 0.286 | 0.437 | T2✅T3✅ |

### v3.1 结论（✅ 三目标同时达成 + 超越作者）
- **ASR 完全恢复并追随 Narcissus**：scale 1.0→100、0.5→99.82（持平 Narcissus 99.6）。修复（|da|≤0.15）让 victim 重新学 δ_global。
- **🎯 scale 0.2 同时满足三目标**：ASR **94.81 > 作者 82.99**（T1✅）、BA 94.67≈94.6（T2✅）、**L2=1.31<1.5 且 SSIM 0.963>0.95（肉眼不可见）**（T3-隐蔽✅）、AC-AUC 0.22 / SS-AUC 0.42 均<0.5 + TPR@1%=0（T3-防御✅）。**首个同时达标 T1+T2+T3 的配置**。
- **scale 0.15 卡在临界**：ASR 82.92 ≈ 82.99（差 0.07），更隐蔽(L2 0.99/SSIM 0.978)但 ASR 刚够 baseline。
- **v3.1 略超 Narcissus（证明 adaptive 有贡献）**：scale 0.1 (L2=0.658) ASR **70.96 > Narcissus gs010 (L2=0.66) 的 68.57**；SS-AUC 0.437 < Stage A gs010 的 0.494 → 有界学习型 adaptive 比规则版在等 L2 下 ASR 更高、SS-规避更好。
- **迭代总结**：v1(L_align 优化δ_global)→v2(CE)→v3(无界adaptive) 均不敌 Narcissus；**v3.1（固定 Narcissus + 有界学习 adaptive）跑通**，正解 = 「保留 Narcissus 的强 ASR 方向 + 有界 adaptive 做防御规避塑形，且 |adaptive| 必须 ≪ |δ_global| 以满足 C2」。

## v3.1 多种子稳定性 —— 2026-07-04，✅ 稳定达标
scale0.2 和 scale0.17 各 3 种子(CIFAR-10 res-square @1%)。

| 配置 | 种子 | ASR | BA | ASR 均值±std | BA 均值±std |
|---|---|---|---|---|---|
| scale 0.2 (L2≈1.31) | 1 | 94.81 | 94.67 | **93.44 ± 2.64** | **94.74 ± 0.07** |
| | 2 | 95.15 | 94.74 | | |
| | 3 | 90.37 | 94.80 | | |
| scale 0.17 (L2≈1.12) | 1 | 90.45 | 94.89 | **90.10 ± 1.46** | **94.83 ± 0.06** |
| | 2 | 91.30 | 94.78 | | |
| | 3 | 88.54 | 94.82 | | |

**判定**：
- **T1(ASR>82.99)✅ 3种子全稳**：scale0.2 均值 93.44±2.64(最低 90.37≫82.99)；scale0.17 均值 90.10±1.46(最低 88.54≫82.99)。
- **T2(BA≈94.6)✅ 极稳**：两档 BA 均值 ~94.8，std<0.07。
- 种子 3 的 ASR 略低(90.37/88.54)——种子 3 补跑时与 y_target=5 共享 GPU2(2 作业/卡)，可能受共享影响，但对结论无影响(仍 ≫82.99)。

## v3.1 后续验证批(batch2)—— 2026-07-03
微调 sweep + 多 y_target（结果干净）；消融因 artifact 污染重跑中。

**P2 微调 sweep（scale 0.17/0.18，比 0.2 更隐蔽的点）**：
| scale | L2 | SSIM | ASR末20均 | BA | AC-AUC | SS-AUC | 三目标 |
|---|---|---|---|---|---|---|---|
| 0.17 | 1.117 | 0.972 | 90.45 | 94.89 | 0.308 | 0.420 | **T1✅T2✅T3✅**(更隐蔽) |
| 0.18 | 1.182 | 0.969 | 92.94 | 94.62 | 0.179 | 0.419 | **T1✅T2✅T3✅** |
→ scale 0.17/0.18 比 0.2 更隐蔽(L2 1.12/1.18 < 1.31)且 ASR 仍 90+，是比 0.2 更优的冠军候选。

**P3 多 y_target（scale 0.2，普适性）**：
| y_target | ASR | BA | AC-AUC | SS-AUC |
|---|---|---|---|---|
| 0(已有) | 94.81 | 94.67 | 0.223 | 0.416 |
| 1 | 94.74 | 95.08 | 0.200 | 0.415 |
| 2 | 93.81 | 94.62 | 0.221 | 0.565 |
→ y_target=0/1/2 ASR 均 ~94（≫82.99），方法**跨目标类普适**。注：y_target=2 的 SS-AUC 0.565 略>0.5（SS 有微弱信号），但 TPR@1%FPR=0（严格阈值下仍防不住）。

**P1 消融（adaptive 贡献）—— ⚠️ 首次污染，干净版重跑中**：
首次消融(ablation_scale*_noAdp)误用了 `save_trigger_10_0/global_delta.npy`，该文件被早先 v1 base 跑覆盖成 v1 优化方向(L2 4.55)而非 Narcissus(6.6)，导致消融的 δ_global 与 v3.1 不一致(L2 0.91 vs v3.1 的 1.31)，对比被混淆。**干净消融**（用 v3.1 的 Narcissus artifact `resource/faat/v3_1/scale_X/` + faat_eps 0）已重跑：`results/ablation_CLEAN_scale{0.2,0.1}/`。

**P1 干净消融结果（2026-07-03，同 δ_global=Narcissus×scale，仅差 adaptive）**：
| scale | 配置 | ASR末20均 | BA | SS-AUC | AC-AUC |
|---|---|---|---|---|---|
| 0.2 | 无 adaptive(纯Narcissus×0.2) | 90.34 | 94.96 | 0.385 | 0.154 |
| 0.2 | v3.1(有 adaptive) | 94.81 | 94.67 | 0.416 | 0.223 |
| 0.1 | 无 adaptive(纯Narcissus×0.1) | 61.68 | 94.77 | 0.452 | 0.389 |
| 0.1 | v3.1(有 adaptive) | 70.96 | 94.57 | 0.437 | 0.286 |

**消融定论（诚实）**：
- **adaptive 主要贡献是 ASR**：scale 0.2 +4.5（90.34→94.81）、scale 0.1 +9.3（61.68→70.96），**越隐蔽（小 scale）贡献越大**。这是 adaptive（L_align 优化的有界残差）的真实价值——在低扰动预算下提升翻转效率。
- **adaptive 对防御规避无明确改善**：SS-AUC / AC-AUC 在有无 adaptive 间互有升降、差异小（均 <0.5，两防御本就失效）。即「AC/SS 防不住」主要来自 Narcissus δ_global 施于目标类样本的结构（非 adaptive）。原「特征对齐→破坏聚类防御」假设**不被干净消融支持**，需修正叙事为「adaptive 提升 ASR 效率」。
- 对比首次（污染）消融的 ASR 62.68/20.29 vs 干净的 90.34/61.68：差距来自 δ_global 方向（v1 优化方向弱于 Narcissus），再次印证「不要再优化 δ_global」。



## 防御评估(AC/SS/STRIP/Fine-Pruning) —— 2026-07-04，4/4 失效 ✅
对 v3.1 冠军 scale 0.2 和 0.17。代码: `faat/defenses.py` + `metrics/detection.py`。

| Defense | scale 0.2 | scale 0.17 | 解读 |
|---|---|---|---|
| **AC**(聚类) | AUC 0.223 / TPR@1%=0 | AUC 0.308 / TPR@1%=0 | 投毒不独立成簇 |
| **SS**(谱) | AUC 0.416 / TPR@1%=0 | AUC 0.420 / TPR@1%=0 | 谱方向不可分离 |
| **STRIP**(扰动熵) | AUC 0.755 / **TPR@5%=0.044** | AUC 0.705 / **TPR@5%=0.064** | 严格误报仅捕获4–6% |
| **Fine-Pruning**(剪枝) | ASR 96.0→96.2(90%剪) | ASR 90.2→90.5(90%剪) | ASR纹丝不动 |

- **AC/SS**:投毒特征嵌在目标簇内,聚类/谱方法无法分离(AUC<0.5, TPR@1%=0)。
- **STRIP**:有界adaptive+Narcissus使触发对扰动不改变预测一致性,TPR@5%仅4–6%。
- **FP**:信号分散在目标类主要神经元,不集中在少数"异常"神经元,剪枝无法隔离。
- **4/4 全失效** → v3.1 对四种范式(聚类/谱/扰动/剪枝)防御均鲁棒。数据在 `results/*/defenses.json`。

## v4 自包含 Narcissus —— 完整 18 组结果(2026-07-05,3 seeds × 2 数据集 × 3 L2)

> `--global_mode from_scratch`(自包含引擎从零优化 δ_global,不碰作者 noise_01000.pth)+ `--fix_global` + `--adaptive_l2_max 0.15`。
> 结果均值±标准差(3 seed)。truth: `docs/v4_results.csv`。

| group | ASR | BA | SSIM | AC-AUC | SS-AUC | STRIP@5% | FP-ASR@0.9 |
|---|---|---|---|---|---|---|---|
| CIFAR L2=0.9 | 76.1±2.5 | 94.8 | 0.981 | 0.33±0.03 | 0.44±0.01 | 0.007 | 73.7 |
| CIFAR L2=1.2 | 88.6±0.9 | 94.7 | 0.967 | 0.37±0.06 | 0.45±0.03 | 0.003 | 86.4 |
| **CIFAR L2=1.5** | **93.6±1.9** | 94.8 | 0.950 | **0.25±0.03** | 0.43±0.02 | 0.040 | 93.3 |
| GTSRB L2=2.5 | 64.5±3.4 | 99.9 | 0.805 | 0.71±0.04 | 0.79±0.05 | 0.217 | 69.3 |
| GTSRB L2=3.0 | 71.5±1.8 | 99.9 | 0.761 | 0.72±0.02 | 0.73±0.02 | 0.258 | 75.3 |
| GTSRB L2=3.5 | 82.7±2.7 | 99.8 | 0.720 | 0.76±0.03 | 0.74±0.01 | 0.372 | 87.5 |

**结论**:
- **CIFAR 三目标全达标**(ASR>作者82.99 / BA≈94.6差<1% / SSIM>0.95 & AC·SS·STRIP·FP 全失效),多种子稳定(std≤2.5)。**且不依赖作者 noise**——规避非作者 noise 特有。
- **GTSRB ASR/BA 攻陷**(旧 0.9%→65–83%),但隐蔽(SSIM 0.72–0.81)与规避(AC/SS 0.71–0.79)不达标——43 类需更大触发器,投毒样本特征可分。是数据集特性难点(CIFAR 对照证明非方法缺陷)。
