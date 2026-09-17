# HANDOFF — 双论文线状态交接（2026-09-16，供新会话冷启动读取）

> 本文件浓缩 2026-09-11 至 09-16 全部会话的关键结论、数字与作者决策。
> **新会话第一步：读完本文件，再按需查阅各索引文件。** 不要凭记忆重建任何数字。

---

## 0. 一句话现状

训练阶段收官（表征线 51/51 run 完成），结论已定型且自洽；**下一步是写作与两线各自的收尾**。战略决策已定：**FAAT 与表征工作分开写两篇论文**。

---

## 1. 两条资产线（两个仓库）

| | 仓库 | 状态 | 内容 |
|---|---|---|---|
| **线 A：FAAT 攻击论文** | `/media/hd1/wangxin/work7-7month/GeneralComponents-main`（**已冻结只读**，tag `v5.0-paper-frozen` @`1b60400`） | 论文已编译（CVPR 格式 8 页 + supplement，`paper/main.pdf`），**待收尾投稿** | FAAT 主方法 4 数据集全超 baseline（+3.9~+51.9 ASR，BA 无损）；KST 隐蔽变体；GTSRB stress test。数字溯源链见其 `docs/BASELINE-FREEZE-2026-09-11.md` |
| **线 B：表征/防御论文** | `/media/hd1/wangxin/work7-7month/ood-calibration`（活跃开发，main 分支） | **训练完成（51 run），待补缺口 + 写作** | clean-label 受限投毒的可学性边界相图 + 表征机制 + 检测解耦。核心文件：`docs/PILOT-OBSERVATIONS.md`（Obs1-6 全文）、`CLAUDE.md`（定位+纪律）、`docs/pilot_metrics.csv`（51 行）、`docs/pilot_representation.csv`（45 行）、`docs/figs_pilot/`、`scripts/analyze_all.py`（一键复算全部汇总表） |

线 A 的备份：日志 tarball 在 `/media/hd0/wangxin/backup/` + GitHub `backup/results-logs` 分支。线 B 的日志同样在 hd0（`ood_*_logs_*.tar.gz`）+ git 逐 run 提交。
**远程（2026-09-17 起）**：两线都推到 `github.com/ByteTitan-star/faat.git`——线 A = `exp/kst-sdt` 分支（论文 `468df37` 已推），线 B = `ood-calibration` 分支（ood-calibration 本地 main 跟踪它，`git push` 即推）。

---

## 2. 战略决策演进（作者意图史，勿回退）

1. **09-11**：评估 work7-7month 可发表性 → FAAT 论文成熟但 novelty 承压。
2. **09-11**：作者选定了当时的新主线"OOD-Calibrated Target-Specific Trigger"（OOD 做显式负校准参照，区别于 Narcissus 的 surrogate 用法）；定 48h 验证纪律 + 预注册 Go/No-Go。
3. **09-12**：48h 结果 **NO-GO**——OOD 负校准作为攻击组件被证伪（GTSRB c 臂 ASR 3.3/4.0 vs a 臂 49.5/50.7）。作者当日决定**转向防御导向表征研究**（可学性边界 + 检测可观测性），OOD 降为实验变量。
4. **09-12**：作者做了第二轮 novelty 核验（15 条修正），定稿表述安全区（见 §4）。
5. **09-16**：训练收官后，作者问"触发器强还是防御强"，并做最终战略决策：**分开两篇**。理由：叙事互斥（攻击 vs 防御表征）、时间线不同步（FAAT 能投、表征还要补）、风险不隔离（FAAT novelty 承压不应拖累表征线）、实验设定不同源（表征用简化触发器族，非完整 FAAT）。
6. **可选的桥（已同意的方向）**：把 FAAT 作为相图中的一个额外触发器族跑进边界图（堵"只扫了一个攻击族"的审稿问题）——这是扩展不是合并。

**三篇论文的叙事弧**：Wicked Oddities（ICLR 2025，投毒哪些样本）→ FAAT（用什么触发器）→ 表征论文（什么时候学得会、什么时候检得到）。

**意图澄清（09-16，作者原话级）**：FAAT **本身就是作者自研的触发器范式**（初心已由 FAAT 实现一次：攻击高效+防御无效两件事 FAAT 论文都证了）。新实验里的 a/b/c/cur 臂是**简化消融触发器（仅 δ_global），不是 FAAT**——讨论"新范式"时勿把两者混淆。初心论文 = FAAT；下一代范式的前沿 = GTSRB 极自信墙（FAAT 在此付隐蔽性代价，是其自身遗留的开放问题）。

---

## 3. 线 B 最终实验结论（51 run，全部可复算）

### 相图（RQ①②）
- **预算轴平坦**：a 臂 17→189 投毒样本 ASR 40.6→50.7；cur 臂（最强触发器）17→189：75.5→84.5。两家族均无 ED50 式阈值。（GTSRB 投毒上限 ≈ 目标类 189 张，pr>1% 会被静默截断——只能向低扫。）
- **校准轴阶跃**：arm c 用 ood_weight 插值，w=0（≡a 臂）10/10 获取（40.6–50.7）；w≥0.05 → **0/18 获取**（6 个 w 值 × 2-3 seed，最高 11.5，零例外）。**L50 < 0.05**。
- **跨 regime 排序不变**：cur ≈ b > a > c 在 CIFAR-10（天花板）/ CIFAR-100 / GTSRB 全部保持。
- **校准严重度随 victim 置信度缩放**：GTSRB（BA 99.9%）致命阶跃；CIFAR-100（BA 77%）−15 税（a 94.0 → c 79.2）；CIFAR-10 混合。
- **schedule 不变性通过**：GTSRB/CIFAR-10 的 a/b/c 在 150ep 与 300ep 排序全保持 → 短 schedule screening 解锁。

### 机制链（RQ③）
- **训练前预测量**：proxy 方向通用度（触发器阶段、不碰 victim 即可测）与 victim ASR 的 Spearman ρ = **0.899**（GTSRB, n=38）/ **0.800**（CIFAR-10, n=9）/ **1.000**（CIFAR-100, n=4）。
- **表征签名**：获取 = 特征偏移集中到少数通道（GTSRB acquired gini 0.714±0.017 n=20 vs non-acq 0.633±0.026 n=15，间距 2.6σ）+ 触发特征并入目标簇（低 sep）；非获取 = 自建远簇（c 臂 sep≈11）+ 偏移弥散。
- layer-wise 1−CKA 质变：cur 臂触发表征与 clean 近正交（0.91）且获取最好；c 臂"温和"偏移但背离锚点。

### 检测解耦（RQ④，量化）
- 同一 AC 检测器对**已获取**（ASR≥20）后门：GTSRB 0.77±0.10 → CIFAR-10 0.25±0.05 → CIFAR-100 **0.02±0.01（反相关）**。
- 反向解耦：GTSRB 非获取触发器仍获 0.59 响应——**检出 ≠ 有害**。
- 机制自洽：越成功的后门越把投毒特征折叠进目标类内部，AC/SS 找独立簇 → 越成功越检不到。

### "触发器强还是防御强"的标准答案（作者 09-16 问，已定稿）
> 中低置信度 victim 上触发器几乎完胜（17 样本获取 + 检测反相关）；样本级防御不是弱是"看不见"；唯一平局在极自信模型（GTSRB），但那是置信度轴挡的、不是防御的功劳——能学的就可检、隐蔽的就学不会（Pareto 僵局）。

### 遗留缺口（写作时如实声明 + 部分待补）
单架构（ResNet18）；CIFAR-10/100 无非获取点（天花板）；w∈(0,0.05) 未加密；检测层仅 AC/SS 样本级（STRIP/NC/FP 待补）；FAAT-as-trigger-family 未跑。

---

## 4. Novelty 禁区（作者两轮核验定稿，违者即错）

**不可 claim**：首次研究 backdoor learnability（Xian ICML 2023 Adaptability Hypothesis）；首次强度↔检测关系（Khaddaj ICML 2023 *Rethinking Backdoor Attacks*，feature strength + AUROC）；首次 target-only / target-only+OOD（Narcissus CCS 2023 就是 target-class + public OOD）；首次投毒率相变（Zheng，Harvard 课程稿、LLM，只作 concurrent motivation）；clean-label 无理论（Yu ICML 2024 generalization bound）。

**安全主张**：系统刻画 target-only clean-label 图像后门的多约束 empirical learnability boundary；OOD 角色/距离/diversity 如何移动边界；learnability–representation–detectability 何时耦合/何时**解耦**（decoupling region 是 RQ④ 靶心）。理论框架对照：Pichler AISTATS 2024（检测不可行性）。Threat-model anchor：Wicked Oddities（ICLR 2025，自己的前作）。

**核心命题（定稿原句）**：*"This project does not optimize a new backdoor attack. It characterizes when clean-label backdoors become learnable under target-only poisoning, how OOD calibration shifts this acquisition boundary, and when learned backdoors become detectable—or remain observationally indistinguishable to existing defenses."*

**其他表达纪律**：c 臂写 "crossed into the non-acquisition regime"，不写 failed；AC/SS 定位为"检测可分性分析"（P3 层），方法目标不得是逃逸防御；"may reduce the separability of poisoned samples in representation space" 措辞。

---

## 5. 实验设计红线与方法学约定

- 防御指标**分层**：AC/SS = training-sample 检测（AUROC/TPR@FPR）；STRIP = test-input 检测；NC = model/class anomaly；FP = mitigation（报 ASR 降幅 vs BA 退化，不报 AUROC）。
- "可学"须多 seed 稳定；边界用 **P_L(x) = P(acquisition|x)** 与 L10/L50/L90，过渡带加密 seed；acquisition 阈值现用 ASR≥20。
- schedule 不变性已验证；短 schedule 可作 screening。
- coarse-to-fine 围绕已知过渡带，不做全网格。
- 三层指标：行为（BA/ASR/方差）、表征（distance/CKA/gini）、防御（分层如上）。

---

## 6. 下一步清单

### 线 A（FAAT，1-2 周内可投，**会议已定 CVPR 2027**）
1. ~~补 `paper/README.md` 列出的 `—` 占位~~ **✅ 全部完成（2026-09-16/17，含 physical-robustness 图）**：T1/T4/T7 无剩余 `—` 格。T1 CIFAR-100 MultiBpp 14.5/17.3（`results/kst_sdt/c100_bl_mbpp*`）、T1 Tiny MultiBpp-RGB/B **62.5/55.0**（复现，`bl_tiny_quantizeB_seed1` 单 seed BA 54.9）、T4 Narcissus s-dprime 0.16（同管线重放，KST 重放 4.6433 逐位验证）、**T4 BppAttack 整行 99.9/0.954/2.83/3.6e4/0.03**（`results/kst_sdt/bppattack_q24_28_8_pr0.05_s1_result.json`，5% dirty、24:28:8、与 KST/Narcissus 行同评估协议；行内 SSIM 用实测 0.954 替换引用 0.97 使整行同源）。supp T7 2 格此前已填。
   **F14 physical-robustness 图已完成（2026-09-16）**：`paper/figs_scripts/gen_fig14_physical_robustness.py` 对 3 个 v4 CIFAR-10 L2=1.5 seed 模型测 JPEG(q10-95)/rotation(±5-30°)/rescale(0.6-1.5) 触发样本的 ASR/BA 曲线（`_data/data_physical_robustness.csv`，重画用 `plot` 参数）。结论：后门与 benign 任务同尺度退化（q95/±5°/s0.8 时 ASR>75%；q30 以下攻击与 BA 同崩）。supplement 新增 Sec "Physical robustness"。**论文已无任何 —/pending 残留**。
   ⚠️ **refs.bib 清查完成（2026-09-16/17，verify 全清）**：9 条错误引用已修正——Narcissus 实为 **CCS 2023**（标题 "with Limited Information"，原作者名单全错已改）；BppAttack 实为 **CVPR 2022 Wang et al.**（bib 原作者"Bui Arman"纯属错误，key 改 wang2022bppattack）；I-BAU 实为 ICLR 2022 "Adversarial Unlearning of Backdoors via Implicit Hypergradient"（key 改 zeng2022ibau）；RNP 实为 ICML 2023 "Reconstructive Neuron Pruning"（key 改 li2023rnp）；DeepInspect 实为 **IJCAI 2019** Chen/Fu/Zhao/Koushanfar（原条目作者/venue/年份全错）；generalcomponents2025 补全 Wu, Zhixiao 等 6 作者 + NeurIPS 2025 + 全标题；BAAT/COMBAT/NoiseAttack 全名单核实（key 改 miah2024noiseattack）。tex 引用已 sed 同步，两份 PDF 0 undefined。
   ⚠️ **bl_tiny 真相（2026-09-16 二次核实）**：先前"9 个 bl_tiny run 无效（BA 14%）"是**解析 bug**（CleanLoss 列误读为 CleanACC）。全部有效（BA≈55），正确末20均 3-seed：BadNets-C **89.3±3.6** / Blended-C **79.3±1.4** / MultiBpp-RGB **62.5±1.7**（32×32 crop 管线、res-square、0.25%）。远高于 T1 引用的原论文 39.0/43.9（64×64+9×9 patch 管线，不同设定不可比）。**作者决策（2026-09-16）：T1 Tiny 行 BadNets/Blended 保留引用数字**（headline +51.9 不变），caption 已明确披露 cited vs reproduced 与管线差异；matched 复现数字存档于 `paper/README.md` 供审稿回应；Limitations 同步。
2. ~~核实 ablation_CLEAN 对照疑点~~ **✅ 已核实（2026-09-16）**：`ablation_CLEAN_*` = **干净版无 adaptive 消融**，不是 clean 对照（"CLEAN"指修复了首次消融 `ablation_scale*_noAdp` 的 δ_global 污染）。配置实锤：poison_rate=0.01（投毒 500 张）+ faat + faat_eps=0 + 加载 `resource/faat/v3_1/scale_X` Narcissus 触发器；日志复算末20均 PoisonACC 61.68/90.34 与文档一致。**真正要弃用的是旧 Table 6 的 no-Adp 20.3/62.7（污染数据）**；论文 `tab_ablation.tex` 本来就用对了（δ_global only 90.3 行，SSIM 0.963/L2 1.313 已独立复算吻合）。`all_result.md`/`docs/paper_tables.md` 错误记录已就地更正（保留原文痕迹）；
3. **FAAT novelty 冲突地图（2026-09-16 查证）**：
   - 🔴 δ_global ↔ Narcissus（CCS 2023）：直接冲突面。Narcissus 核心机制就是 optimized universal noise；v3.1 用其 artifact、v4 同目标重优化（代码注释自认 "precisely the Narcissus objective"）。related work 必须第一段正面区分，定位句：**"通用方向触发器在困难 regime 崩溃（自家 GTSRB/大类数据证明），FAAT 的 adaptive+alignment 机器让该范式在 4 数据集活下来并规避防御——贡献是机器+证据，不是方向本身"**；
   - 🟡 L_align ↔ 质心对齐家族（ESWA 2025 sample-customized feature alignment、FFCBA arXiv 2504.21054、Zeng/Luo/Ma 特征空间优化线）：自家消融证明 L_align 非 ASR 核心（去掉仍 96.4%）→ **不做 novelty 主张**，定位 defense-shaping 组件；
   - 🟢 δ_adaptive（冻结全局+有界逐图残差+预算耦合）= novelty 承重墙。⚠️ 支撑数字已修订（2026-09-16 核实）：干净消融下 adaptive 贡献 **+9.3（scale0.1：61.7→70.96）/ +4.5（scale0.2：90.3→94.81）**，越小的扰动预算贡献越大——正确定位是"低预算下的 ASR 效率组件"，~~"去掉崩到 20.3"~~（20.3 来自污染消融，勿引用）；guidance 主导时 adaptive 冗余（gs1.0+noAdp 仍 100）。
   - ✅ **δ_adaptive 专门查重已完成（2026-09-16）**：ISSBA（隐写 encoder 全逐图、poisoned-label）/ BAAT（TDSC 2025，语义属性触发器、clean-label）/ COMBAT（AAAI 2024，generator 交替训练）/ Luo 2206.04881（two-phase image-specific，最近邻）/ CVPR2020-video（universal adv trigger，无逐图部分）/ FIBA（频域 universal）/ TDSC 2025 color-space（正交视角）——**全部已发表工作在测试期触发器都是逐样本或 universal 的单层设计，无"冻结全局测试期 + 训练期专用有界逐图残差"的分解先例**。差异句已写入 `paper/sections/02_related.tex`（sample-specific 段）+ 4 条新 bib（zhu2025baat/huynh2024combat/luo2022twophase/nguyen2024noiseattack，部分带 verify 标记待清查）；KST 段已加 NoiseAttack 区分（universal+结构平坦谱 vs 逐样本）。论文已重编译通过（9 页，0 undefined）。
   - KST 部分照旧加 NoiseAttack 区分；refs.bib verify 标记清查；
4. ~~加 Limitations 节~~ **✅ 已完成（2026-09-16）**：`paper/sections/09b_limitations.tex`（单架构 / CIFAR-100+Tiny 引用格与 bl_tiny 废弃 run 如实声明 / 4 防御覆盖 / GTSRB 隐蔽代价），编译通过正文仍在 8 页内；README 数据完整性政策已同步。**会议已定（2026-09-16 作者拍板）：CVPR 2027**（≈2026-11 中截稿，格式已就绪）。

### 线 B（表征，补缺口→写作）
1. **cross-architecture 验证**（最优先——单架构是最大 caveat）：**✅ 行为层+表征层全部完成（2026-09-17）**：VGG16 a/c/cur × 3 seed × 300ep（`results_ood/oodabc_cifar10_{a,c,cur}_seed{1,2,3}_vgg16/`）。
   **行为层：排序不变性跨架构坐实且 seed 稳定——cur > a > c 在 3/3 seed 严格成立**：cur **85.8±2.1** > a **77.9±3.2** > c **66.9±2.9**（BA 均 92.7-93.1 无损）。c 臂税 VGG16 = −18.9，比 ResNet18 的 0/18 全灭阶跃**变浅**——OOD 负校准的"致命性"幅度部分依赖架构，但方向/排序普适。写作句式："the ranking cur > a > c is preserved across architectures (3/3 seeds), though the calibration cliff is softer on VGG16"。
   **表征层（`scripts/pilot_representation_vgg.py` → `docs/pilot_representation_vgg16.csv`，9 行）**：机制签名跨架构保持——**1-CKA(stage5) 排序 cur 0.881 > a 0.770 > c 0.679**，cur 触发表征与 clean 近正交（ResNet18 上 0.91，同签名）；c 臂"温和偏移但背离锚点"复现（mp5 最低 0.679）。gini 三臂无分离（0.58-0.60 重叠）——与 ResNet18 的 CIFAR-10 同因（CIFAR-10 无非获取组，gini 是 acquired vs non-acquired 签名，GTSRB 才可测），**非架构差异，机制自洽**。VGG16 penultimate=fc_2(128d)，conc_top3pct 按同比例（top4/128）口径可比。
   代码改动已提交（`fafd02e` + `2eae011`）；表征脚本与 CSV 待随下次提交入库。触发器优化阶段 proxy 仍是 resnet18（合理 surrogate 迁移设定）。
2. 检测层补全：STRIP（`faat/defenses.py:strip_detection`）、NC（未实现，需写）、FP（`defenses.py:fine_pruning_defense`）；
3. 可选：FAAT-as-trigger-family 进相图（用冻结仓库配方）；
4. 写作：相图主图（预算×校准×P_L）+ 机制链图 + 检测解耦散点；会议定位 S&P/USENIX/CCS analysis 风格或 ICLR/NeurIPS。

---

## 6.5 初心对齐路线 —— 范式 #3 侦察（2026-09-16 定稿，作者批准）

**核心认知**：表征科学不是对初心的偏离，而是给范式设计造了加速器——proxy 通用度预测量（ρ=0.9）让任何新触发器候选可在**不训练 victim 的前提下几分钟预筛**（OOD 校准那次花 48h 是因为直接上 victim；今后"触发器阶段筛 → 存活者才配 victim"）。

**三条线并行**：

1. **线 1：FAAT 收尾投稿**（初心论文 #1 兑现）。其 related-work 定位句同时为范式 #3 立靶：FAAT 自己承认在 GTSRB 付隐蔽性代价（L2=3.5、AC 0.77）= 下一篇的入口。加 δ_adaptive 专门查重。
2. **线 2：表征论文写作**，加一节 **Design Implications**（范式 #3 的设计委托书）：① 通用度是唯一杠杆（ρ=0.9）；② 预算不设防（17 样本够）；③ 折叠进目标簇 = 检测结构性盲区（AC 反相关 0.02-0.25）；④ 极自信模型是唯一 Pareto 墙（能学的就可检、隐蔽的就学不会）。
3. **线 3：范式 #3 侦察（两个候选，一周内出判据）**：
   - **Scout A：目标类主成分方向触发器**（PA-ICT 正参照范式推进）。不用优化噪声方向（Narcissus 系），改用目标类自己的主成分特征方向作 δ——正参照、只依赖目标类、威胁模型不变。依据：PA-ICT 纯 PCA 在旧仓库 97.3 ASR + 绕全图 NC（CIFAR-10），但从未上过 GTSRB。预筛：构造 PCA 方向 δ → 测 GTSRB clean proxy 通用度 → 对比 cur 臂的 0.87。
   - **Scout B：轨迹感知触发器**（新设计轴）。现有方法全对静态 clean proxy 优化 δ；但自家数据显示 GTSRB 上 KST 在 ep139 冲到 19.3% 再被后期自信拟合压灭（PoisonLoss 9.44）→ victim 训练轨迹本身在修剪触发特征。新轴：用 20-30 epoch 短 surrogate 训练轨迹作优化景观，让 δ 学会"活过修剪期"。查新初判：对训练轨迹/剪枝动力学优化未见先例（写作查重时确认）。预筛：短轨迹 proxy 通用度 vs 静态 proxy 通用度，差值即信号。

**预注册 Go/No-Go（沿用 48h 纪律）**：Go 信号 = **在更小 ‖δ‖ 下达到与 cur 臂同等通用度**（= 同 ASR 换更好隐蔽 = 打破 Pareto 的签名）；触发器阶段筛不赢 → 不上 victim 直接砍。防御层沿用现有套件；成活者走 full grid → 范式 #3 论文（攻击第一视角，初心论文 #2）。

## 7. 基础设施索引（线 B）

- 启动：`scripts/run_ood_abc.sh <cifar10|cifar100|gtsrb> <gpu> <seed> <a|b|c|cur...>`，env：`EPOCHS/EPOCH_TAG/POISON_RATE/PRTAG/OOD_WEIGHT/WTAG`
- 四臂语义：a=target-only 正对齐 / b=OOD 混入 pool / c=OOD 负校准（weight 可调）/ cur=非目标类池（同管线锚点）
- 监控：`scripts/watch_ood_abc.sh`（51 tags，自动 STATUS.md + hd0 备份）；看板 `results_ood/STATUS.md`
- 分析：`scripts/pilot_metrics.py`（行为+proxy 通用度）→ `docs/pilot_metrics.csv`；`scripts/pilot_representation_ext.py`（gini/CKA/增广）→ `docs/pilot_representation.csv`；`scripts/analyze_all.py`（五表汇总）；`scripts/pilot_figures.py`（图）
- 汇总文档：`docs/PILOT-OBSERVATIONS.md`（Obs1-6）、`docs/ood_abc_results.md`（NO-GO 判定史）、`CLAUDE.md`（纪律全文）

---

## 8. 已知坑（重踩过，勿再踩）

1. train_backdoor 日志按 seed 命名：seed2/3 是 `output_2/3.log`，不是 `output_1.log`；
1b. **日志解析必须锚定全列**（2026-09-16 教训）：列序 = epoch/lr/time/TrainLoss/TrainACC/PoisonLoss/**PoisonACC(第7列=ASR)**/CleanLoss/**CleanACC(第9列=BA)**。用部分正则会把 CleanLoss 当 BA——曾因此误判 9 个 bl_tiny run "BA 14% 废弃"，实际全部有效（BA≈55）。任何新解析先交叉验证一个已知 run；
2. 多进程**不得**追加写同一日志文件（互覆写，丢输出）；
3. Bash 后台链 `cd X && nohup ... & cmd2` 的 `&` 会把 cd 困在子壳——统一用 `setsid nohup bash -c "cd ABS && ..."` + 绝对路径日志；
4. GPU 与他人共享：只占 ≥4GB 空闲的卡；瞬时 CUDA fault（unspecified launch failure）发生过 3 次，失败 run 查启动器日志后重发即可；
5. `results_ood/` 被 gitignore（备份走 tarball+分支约定）；`pkill -f` 会匹配到自己的包装 shell，用 `^bash scripts/...` 锚定；
6. conda env 无 activate 脚本，直接用 `/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python`；
7. 旧仓库 `GeneralComponents-main` 已冻结，只读引用，不写回。
