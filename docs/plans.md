# FAAT 研究计划（Feature-Aligned Adaptive Trigger）

> 权威计划文件。conda 环境 `.conda-envs/GeneralComponents`（python3.8 + torch1.11.0+cu113）；工作目录 `GeneralComponents-main/`。
> 关联：源设想 `docs/development.md`；**baseline 单一真相源 `docs/result_all.md`**；复现调度 `scripts/run_table1.sh`；Phase 0 结论 `docs/PHASE0-REPORT.md`；Week 9 门控 `docs/GATE-CIFAR10.md`。

---

## 执行现状对齐（2026-07-01，动手前必读）

- **baseline 复现管线已存在且在跑**：`docs/result_all.md` 是 baseline 的**单一真相源**；`scripts/run_table1.sh` 正后台复现论文 Table 1（CIFAR-10, 1%, 8 选择 × 4 攻击 = 32 组）。进度（2026-07-01）：MultiBpp-RGB(`quantize 24:28:8`) 8/8 ✅；**Badnets-C 7/8、MultiBpp-B(`quantize 255:255:8`) 7/8 🔄**；**Blended-C 🔄 0/8 跑中**（用户已补丁 `utils.py` 行187/303 的 blend `checkboard` 由 `[2,1,3]`=0.2:0.1:0.3 改为 `[2,2,2]`=0.2:0.2:0.2，匹配论文 Table1 默认；NAR 行424 未动）。
- **Phase 0 baseline 与复现管线对齐，不另起炉灶**：BadNets / MultiBpp-RGB / MultiBpp-B / Blended 由 `scripts/run_table1.sh` 产出（见 §3.1）。**SIBA / Narcissus / FTROJAN-style 为额外 baseline**（论文外、FAAT 计划需要的对照，尤其 SIBA 是头号对照），GPU 空闲后补跑。
- **数据集顺序门控（用户硬约束）**：第一阶段**仅 CIFAR-10**；仅当 CIFAR-10 上 FAAT 综合超 baseline 后，才依次扩 GTSRB → Tiny-ImageNet → CIFAR-100。
- **已完成基建（2026-07-01）**：
  - `train_backdoor.py` 末轮存 `model_last.pth`(≈44MB) + `args.json` + `poison_inds.json`（additive，try/except 包裹，**不动 ASR/BA 日志列**，`parse_detail.py`/`scripts/run_table1.sh` 不受影响；其后续启动的 run 会自动产出 checkpoint）。
  - 新建 `metrics/`：`stealth.py`(SSIM/L2/Linf/DCT-L1，零依赖) + `detection.py`(AC/SS + 手写 ROC AUC + TPR@1%FPR，纯 numpy) + `smoke_test.py`。2-epoch 烟测通过（`results/phase0_smoke_quantize/`，junk，可删）。
- **Stage A 规则版 FAAT（proxy-free）已实现并 CPU 验证**：`faat/{image_stats,rules,apply_trigger}.py` + 接入 `train_backdoor.py`（`--backdoor_type faat`、`--faat_global_scale`、`--faat_eps`）。`δ_global`=Narcissus noise（测试触发器，C2），`δ_adaptive`=基于纹理的 DCT 低中频噪声（训练专用）。CPU 自测 + wiring check（MyDataset→DataLoader→ResNet18 前向）通过。**实测发现**：① Narcissus noise 全量 L2=6.6/absmax=0.125/SSIM=0.933（**不隐蔽**——stealth 需调小 `--faat_global_scale` 或 Stage B 重优化 δ_global 预算）；② 真实 CIFAR-10 纹理(Laplacian var)中位数 0.066，已据此校准 sigmoid（thr=0.066, slope=0.041；原 0.02 在真实图上饱和到 t≈0.99）；③ 95% 谱能在 band 0，故 `_BASE_BANDS` 避开 band 0、取低中频 1–2；④ δ_global 主导（adaptive eps 0.01–0.04 ≪ 6.6）→ H3 对比须用「**等总 L2 预算**下 global-only vs global+adaptive」设计，否则 adaptive 被淹没。**待 GPU**：≥50-epoch 真实训练验证 ASR>Random（2-epoch 不足以验证 clean-label ASR，见 result_all.md 收敛 epoch 13–19）。
- **已知代码错配**：SIBA 代码读 `./resource/save_trigger_10_0/`（`utils.py:248`），实际触发器在 `./resource/siba/save_trigger_10_0/{uap,mask}.npy`。跑 SIBA 前需 `ln -s` 或给 `train_backdoor.py` 加 `--save_trigger` 覆盖。
- **GPU 规则（CLAUDE.md）**：4×3090 与他人共享；只用空闲显存≥4GB 的卡，**不抢占他人正用的卡**。当前 4 卡被 `scripts/run_table1.sh` 占满 → 暂停启动训练，先做不需 GPU 的代码工作（Stage A 规则版 FAAT）。
- **命名约定**：遵循 `result_all.md` —— `results/<attack>_<selection>[_<res_sel>]/output_<seed>.log`。

---

## 0. Context（为什么做、基于什么做、锚定决策）

**要解决的研究问题**：把后门攻击的研究痛点从「调参难」（中毒比例/触发器强度/混合权重）提升到「机制缺陷」。现有干净标签攻击在「触发器空间—样本选择—触发器设计」三者之间存在两个根本不足：（1）固定触发器在异质样本上无法同时满足高 ASR、低语义破坏、强防御规避；（2）样本选择与触发器设计被孤立处理，触发器不随样本特征自适应。

**基于的代码资产（已确认，非抽象）**：工作目录即 NeurIPS 2025 论文 *GeneralComponents*（arXiv 2509.19947）的官方实现 `GeneralComponents-main/`，是一个**干净标签后门框架**。已具备：
- 5 种攻击实现：`badnets / blend / quantize(=MultiBpp-RGB/BppAttack) / narcissus / siba(样本特定)`，见 `train_backdoor.py:87`、`utils.py`。
- 样本选择指标体系 `get_stats`（`utils.py:480`）+ 预计算 `resource/save_metric_10_res/*.pkl`。
- `UnetGenerator`（`cifar_resnet.py:11`，3→3，Tanh，当前是死代码，可复用）。
- DCT 工具 `dct_2d/idct_2d`（`dct.py:85/99`）。
- `ResNet.extract_feature`（`cifar_resnet.py:160`，512 维）。
- 隐蔽性度量 GMSD（`utils.py:202`，依赖 cv2/scipy，已装）。
- **CIFAR-10 @1% MultiBpp-RGB 的 8 种选择策略已复现完成**（Res-x² ASR 82.99 vs 论文 83.88，BA≈94.6），见 `docs/result_all.md` §3 / `docs/CIFAR10-RESULTS-claudecode.md`。
- 7 种模型（ResNet18/VGG16/MobileNet/DenseNet/AlexNet/GoogLeNet/SqueezeNet）。硬件 4×RTX3090，CUDA11.3。

**四项锚定决策（已与用户确认）**：
1. **主推进目标 = 标准论文版**：完整 FAAT（规则版→可微策略版）+ 核心 baseline + 消融 + 防御评估，10–12 周。
2. **Agent 路线 = 可微摊还策略网络（非 RL）**：遵循 `development.md` 的决定，用小型可微策略网络（hypernetwork）端到端梯度优化，把 Narcissus 式逐样本 bi-level 优化摊还为一次前向。诚实定位为「学习型自适应触发器策略」，而非 RL Agent。
3. **防御评估 = 完整 BackdoorBench 集成**：按 README 方式把 FAAT 的投毒样本索引接入 BackdoorBench，跑主流防御。
4. **数据集 = 顺序门控**：第一阶段**只在 CIFAR-10 上验证**；只有当 CIFAR-10 上 FAAT 的综合指标**已超过 baseline**后，才依次扩展 GTSRB → Tiny-ImageNet → CIFAR-100。不得并行铺开。

**三条贯穿全程的代码层硬约束（来自源码核查，必须遵守）**：
- **C1 注入是 eager 的**：触发器在数据集构建时一次性施加（`train_backdoor.py:241-260`），不在训练循环里。FAAT 需遵循同一契约 `Add_Clean_Label_Train_Trigger_X(...) -> [(img,label,is_poison)]`。
- **C2 测试触发器必须 index-independent**：`Add_Test_Trigger_*` 遍历测试集时没有逐样本信息（`utils.py:40-73`）。**FAAT 的测试触发器只能是共享的 `δ_global`**；`δ_adaptive_i` 只参与训练塑形。这决定 ASR 由 `δ_global` 主导，优化目标必须保证 `δ_global` 强可学习。
- **C3 增强在注入之后**：`MyDataset` 的 `train_transform`（RandomCrop/Flip）在投毒之后施加（`train_backdoor.py:261`、`utils.py:34`）。空间局部化的 `δ_adaptive` 会被随机裁剪/翻转部分解耦 → 缓解：让 `δ_global` 分散而非局部，把 `δ_adaptive` 限制在鲁棒低频带。

**「干净标签」约定澄清**：本仓库的 `Add_Clean_Label_Train_Trigger_*` 实际把投毒样本**重标为目标类**（严格说是 dirty-label），「clean-label」在此指「触发器不可见」。FAAT 沿用同一约定以保持与 baseline 可比；如需严格 clean-label（保留原标签）作为附加实验，单独加 flag，不影响主线。

---

## 一、研究问题重构

### 1.1 这项研究到底要解决什么机制缺陷？
两个**可证伪**的机制命题（不是「调参」）：
- **M1（触发器-样本失配）**：固定全局触发器在异质样本上无法同时满足高 ASR / 低语义破坏 / 低可检测性；存在**样本级 Pareto 异质性**——同一触发器对有的样本高效且隐蔽，对有的样本要么无效要么高度可检。
- **M2（选择与触发孤立）**：把样本选择与触发器设计解耦，会错失「样本特征几何 ↔ 触发器参数」的联合优化增益；触发器应随样本的纹理复杂度、频谱结构、相对目标类特征中心的距离自适应。

### 1.2 与已有工作的创新点（必须用实验守住，否则不成立）
> ⚠️ 最大风险：sample-specific / 自适应触发器**已有大量先例**——SIBA（本仓库就有）、WaNet、Input-Aware（Nguyen & Tran 2021）、LIRA、BppAttack/Quantize（本仓库）、Narcissus（本仓库 resource）、FTROJAN（频域）。FAAT 不能只是「又一个自适应触发器」。

可守住的三层创新（每层都要在主实验/消融里被证据支撑）：
- **C-机制**：首次在干净标签攻击中显式优化「投毒样本特征 → 目标类特征中心」的对齐（`L_align`），以此**机制性地破坏聚类式防御**（AC/SS 依赖投毒样本形成异常簇）。这是区别于 SIBA/Narcissus 的核心——它们不显式对齐特征几何。
- **C-框架**：把 GeneralComponents 的「样本适配触发器」推进为「**触发器进一步适配样本特征几何 + 样本选择联合优化**」，并以**可微摊还策略网络**把昂贵的逐样本 bi-level 优化摊还为一次前向（相对 Narcissus 的逐样本优化是效率/可扩展贡献）。
- **C-分析**：用「样本级 Pareto 异质性」框架统一解释为何固定触发器在异质样本上失败，并给出自适应触发器的收益上界分析（经验性）。

### 1.3 「Agent」是否真的必要？
- **RL Agent：不必要且有害**。`development.md` 已因 4×3090 成本、稀疏反馈、复现性放弃 RL，决定正确。本计划遵循。
- **可微摊还策略网络：必要且足够**。它解决一个非它不可的问题：**把逐样本 bi-level 优化摊还为单次前向**。Narcissus 对每个目标类要重跑一整套优化，FAAT 的策略网络一旦训练好即可对新样本即时生成策略参数。这是「Agent」真正有价值的形态——**不是调参器，而是把昂贵优化摊还的泛化器**。论文表述用「amortized adaptive trigger policy」比「Agent」更诚实、更不易被审稿人攻击。

### 1.4 命名是否合适？
- **保留 FAAT = Feature-Aligned Adaptive Trigger** 为主名。理由：`development.md` 已从「Frequency-Aware」退到「Feature-Aligned」以避开 FTROJAN 的频域首创权冲突，频域只作技术手段。✅
- **不建议主推 A-FAAT /「Agent-Guided」**：会引来「Agent 必要性」质疑。如要强调学习模块，正文用「amortized policy」，标题保持 FAAT。

### 1.5 定位
定位为**攻击方法 + 机制分析**：主线是新的干净标签攻击 FAAT；用「样本级 Pareto 异质性」机制分析作为 motivation 和解释性贡献。不做攻防评估框架（那是 BackdoorBench 的事）。

---

## 二、最小验证实验设计（Phase 0，必须最先做）

> 原则：**先用已有代码、在 CIFAR-10 上、用最小代价证伪/证实核心假设**，再决定是否进入 FAAT 实现。Phase 0 几乎不写新攻击代码，只复用现有 5 种攻击 + 加 2 个轻量防御 + 加几个度量。**注意：BadNets/Blend/Quantize 的复现由 `scripts/run_table1.sh` 承担（见 §3.1），Phase 0 在其产出的 checkpoint 上算隐蔽/检测度量即可。**

### 2.1 验证哪些假设（每个都可证伪）
| 假设 | 陈述 | 证伪则 |
|---|---|---|
| H0 负向动机 | 固定全局触发器存在 ASR↔隐蔽↔检测的硬权衡，且在异质样本上 Pareto 次优 | 自适应方向不成立，转向分析型论文 |
| H1 样本异质性 | 固定 L2 预算下，触发器效率（能否翻转 + 隐蔽性）在样本间强方差 | 自适应收益小，降级 |
| H2 特征对齐助规避 | 把投毒特征拉向目标类中心可降 AC/SS 检测率而不显著伤 ASR | FAAT 核心 C-机制 不成立，重设计 |
| H3 自适应>全局 | 即使规则版自适应，也能在等扰动预算下 Pareto 优于全局静态触发器 | FAAT 价值存疑 |

### 2.2 数据集 / 模型 / 设置（最小）
- 数据集：**仅 CIFAR-10**（已有，已复现）。不下载其他。
- 模型：ResNet18（`cifar_resnet.py:234`）。
- 设置：all-to-one，`y_target=0`；`poison_rate` ∈ {0.01, 0.05}；seed=1（先单种子，后续补 3 种子）。
- 复用 `resource/save_metric_10_res` 的选择指标，跳过 `cal_metric.py`。

### 2.3 Baseline（复现管线 + 额外）
| Baseline | 来源 / 代码 | 角色 / 证明的痛点 |
|---|---|---|
| BadNets（固定 patch） | `scripts/run_table1.sh`（Badnets-C，`--type 0:0:0`） | 全局静态 RGB 触发器，H0 的代表 |
| Blended | `scripts/run_table1.sh`（Blended-C，**待透明度定夺**） | 全局静态混合触发器 |
| MultiBpp-RGB / MultiBpp-B | `scripts/run_table1.sh`（`quantize 24:28:8` 已完成 / `255:255:8` 训练中） | 干净标签自适应（前身），FAAT 要超越 |
| SIBA（样本特定） | 额外（`--backdoor_type siba`，**需修路径**） | **关键对照**：样本特定但**不特征对齐** |
| Narcissus | 额外（`--backdoor_type narcissus`） | 逐样本优化的干净标签 |
| FTROJAN-style（频域） | 额外（小新代码：`dct_2d` 低频正弦扰动 <50 行） | 频域 baseline，回应「频域首创权」 |

### 2.4 如何验证「RGB 空间不可分性」（= H0/H1，术语见 §1.1 修正）
1. 跑 BadNets/Blended 一组**强度梯度**（Blended 的 `blend_size` 与 alpha 阶梯；BadNets 的 alpha 阶梯），记录每个强度下的 (ASR, L2, SSIM, GMSD, AC-检测率, SS-检测率)，画 **ASR–检测率 Pareto 曲线**。H0 成立 ⇔ 该曲线存在明显权衡且无点能同时占优。
2. **样本异质性证据**：对 SIBA/Quantize 训练好的模型，用 `extract_feature` 取每张投毒样本的特征，算 `||f(x')−c_target||` 与「该样本是否被成功翻转」的逐样本关系，画散点 + 方差。H1 成立 ⇔ 翻转成功率/隐蔽性在样本间显著方差，且与样本特征（纹理/频带能量）相关。

### 2.5 如何验证「样本选择与触发器设计孤立性」（= H3）
- **最小自适应对照**：写一个 ~80 行的**规则版自适应触发器**（§4.7 Stage A 的极简版：纹理复杂度高→高频带+大 ε；平滑→低频带+小 ε；`δ_global` 用现有 Narcissus noise），与「相同 ε 预算的全局静态触发器」对比 ASR–检测 Pareto。H3 成立 ⇔ 规则自适应在等预算下 Pareto 占优。

### 2.6 必需指标（现有 + 新增轻量，已建好 `metrics/`）
- 攻击/效用：ASR（PoisonACC）、BA（CleanACC）——已有（`parse_detail.py`）。
- 隐蔽：L2、L∞、**SSIM（已写 `metrics/stealth.py`）**、GMSD（已有 `utils.py:202`）、**DCT 频谱 L1（已写，用 `dct_2d`）**。
- 检测（已写 `metrics/detection.py`，复用 `extract_feature`）：**Activation Clustering (AC)**、**Spectral Signature (SS)**——报 TPR@FPR=1% 与 AUC。
- *LPIPS 暂不做*（需装包，留到 Phase 2）；Phase 0 隐蔽性用 SSIM/GMSD/DCT-L1 足够。

### 2.7 预期结果与「不符合预期」的含义
- **预期**：H0 成立（全局静态有硬权衡）；H1 成立（样本间强方差）；规则自适应（H3）Pareto ≥ 全局；特征对齐方向（H2 初判）在 AC 检测上可见差异。
- **若 H0/H1 不成立**（异质性弱、全局触发器已能占优）→ **停止 FAAT 攻击路线，降级为「干净标签后门权衡机制分析」论文**（见 §9.6、§11.1）。这是合法且有价值的产出，不是失败。
- **若 H2 不成立**（特征对齐不降检测）→ 重新设计 `L_align` 层位或换对齐目标（如对齐到「目标类决策边界法向」），再验证。
- **Phase 0 通过门**：H0∧H1 成立 + H3（规则自适应）至少持平全局且在检测侧有改善 → **进入 Phase 1（规则版 FAAT）**。

---

## 三、Baseline 复现路线

### 3.1 复现顺序（与 `scripts/run_table1.sh` 对齐，不一次铺开）
- **第 0 批（已完成）**：MultiBpp-RGB(Quantize) × 8 选择策略（CIFAR-10 @1%）。✅ 见 `docs/result_all.md` §3。
- **第 1 批（`scripts/run_table1.sh` 进行中）**：Badnets-C、MultiBpp-B × 8 选择策略；Blended-C 暂缓（透明度定夺）。**由复现管线承担，不重复跑。**
- **第 2 批（FAAT 出结果后，GPU 空闲）**：FTROJAN-style 频域 baseline；SIBA（修路径）、Narcissus；+ Input-Aware / WaNet（若时间允许，作强对照）。
- **可放弃**：Random/Loss/Gradient 等弱选择策略（复现记录已显示其高方差），主表只保留 Res-x²（强）+ Forget（SOTA 选择）+ Random（下界）。

### 3.2 每个 baseline 的作用与证明的痛点
| Baseline | 作用 | 证明/对照的痛点 |
|---|---|---|
| BadNets / Blended | 全局静态触发器 | H0：RGB 静态的权衡缺陷 |
| MultiBpp-RGB / MultiBpp-B | 干净标签自适应（前身） | 「样本适配触发器」的天花板，FAAT 要超越 |
| SIBA | 样本特定、不特征对齐 | **最关键对照**：证明 FAAT 的特征对齐 ≠ 仅仅样本特定 |
| Narcissus | 逐样本优化干净标签 | FAAT 摊还策略的效率对照 |
| FTROJAN-style | 频域触发器 | 回应频域首创权，证明 FAAT 非仅频域 |
| Input-Aware / WaNet（增强） | 输入条件/不可见触发 | 强对照，堵「自适应已有」的审稿意见 |

### 3.3 每个 baseline 必须记录的结果
固定字段（写入 `results/<run>/`，见附录 B）：ASR(末20 epoch 均)、BA(末20 epoch 均)、L2、L∞、SSIM、GMSD、DCT-L1、AC-TPR@1%FPR、AC-AUC、SS-TPR@1%FPR、SS-AUC、训练时长、GPU、seed、完整 args。**所有方法走同一 `train_backdoor.py` 管线**，保证 ASR/BA 列字节级可比（`parse_detail.py` 不改 ASR/BA 列位置）。每跑完一组**必跑 `parse_detail.py --write-md docs/result_all.md` 更新单一真相源**（CLAUDE.md 强制规则）。

### 3.4 公平性保证
- 同一模型（ResNet18）、同一优化器/调度（SGD lr0.1 m0.9 nesterov wd5e-4, milestones[60,90] γ0.1, bs128, 300ep）、同一种子、同一选择指标来源、同一 `poison_rate`。
- 隐蔽性度量对**所有**方法在同一投毒集上算（含全局与自适应）。
- 防御检测对所有方法用同一 AC/SS 实现、同一阈值策略。
- 单种子先跑通，关键结论补 3 种子报均值±方差。

---

## 四、FAAT 方法设计（完整框架，代码级）

### 4.1 整体 pipeline
```
[样本选择] --复用 get_stats/res--> poison_inds (CIFAR-10 类内)
   |
[代理模型 & 特征中心] --ResNet18 干净代理(512d), c_target=类内特征均值--> 冻结
   |
[逐样本特征摘要 s_i] --纹理复杂度/到c_target距离/DCT频带能量/选择先验--> [N, d_s]
   |
[策略网络 π_θ] --MLP hypernetwork, 可微--> 策略参数 (频带权重 w_band, 通道权重 α, 预算 ε_i, mask稀疏 ρ_i, 损失权重 λ)
   |
[触发器生成器 G] --δ_i = δ_global + δ_adaptive_i; DCT带投影+稀疏门--> x'_i = clamp(x_i + δ_i)
   |
[攻击者离线优化] --冻结代理, 反传到 π_θ/G/δ_global, 组合 4 损失--> 保存 artifact
   |
[Eager 注入] --Add_Clean_Label_Train_Trigger_faat--> 走标准 train_backdoor.py 训练+测ASR/BA
   |
[测试触发器] --仅 δ_global（C2 约束）-->
```

### 4.2 「Feature-Aligned」的确切含义
**对齐到目标类的中间层特征中心**（不是频域、不是纹理）。`f=ResNet.extract_feature`（512d，penultimate），`c_target = mean_{x∈clean target class} f(x)`。`L_align = ||f(x')−c_target||₂`（可加 `−cos` 项）。含义：让投毒样本在受害模型的判别空间里**落进目标类簇内**，从而既提高 ASR（更易被判为目标），又使 AC/SS 难以把它从目标类里统计分离。**频域/纹理只是生成器的参数化手段，不是对齐目标**。

### 4.3 「Adaptive」自适应什么
联合自适应（由策略网络 π_θ 输出）：① DCT 频带权重 `w_band`；② RGB 通道权重 `α`；③ 扰动预算 `ε_i`（L2 上界）；④ 空间 mask 稀疏度 `ρ_i`；⑤（Stage C）损失权重 `λ_*`。**位置不显式自适应**（受 C3 增强 + C2 测试约束限制，改用频带/mask 隐式表达空间分布）。

### 4.4 策略网络 π_θ（摊还器，§5 详述）
- 输入 `s_i ∈ ℝ^{d_s}`，`d_s ≈ 12`（纹理1 + 到中心距离/余弦2 + DCT 8 带能量 + 选择先验1）。
- 结构：`Linear(d_s,64)→GELU→Linear(64,64)→GELU→Linear(64,d_out)`，`d_out≈B+3+1+1`。
- 输出经 sigmoid/softplus 范围映射，保证梯度流动且有界。

### 4.5 触发器生成器 G（复用 `UnetGenerator`）
- `δ_global ∈ ℝ^{3×32×32}`：单一可学习全局方向（~3K 参数）。**这是测试触发器（C2），ASR 主力，必须强对齐。**
- `δ_adaptive_i`：**复用** `UnetGenerator`（`cifar_resnet.py:11`，当前死代码），输入 `x_i`，输出 Tanh 残差。
  - `D = dct_2d(δ_adaptive_i)` → 软带掩码 `M_band(w_band)` → 通道乘 `α` → `δ_adaptive_i = idct_2d(D⊙M_band)·ε_i` → 稀疏门 `mask_i=sigmoid((|δ_adaptive_i|−τ_i)/τ)` → `δ_adaptive_i ⊙ mask_i`。
- `δ_i = clip_L2(δ_global + δ_adaptive_i, ε_global)`，`x'_i = clamp(x_i + δ_i, 0,1)`。

### 4.6 攻击者离线优化（`faat/optimize.py`）
- 优化对象：`{δ_global, π_θ, UnetGenerator}`；**冻结**：代理模型、`c_target`。
- 4 损失（§7 定义）：`L = λ_asr·L_align + λ_freq·L_freq + λ_perc·L_perc + λ_div·L_div`（Stage C 加 `L_div`）。
- Adam（π_θ/G lr1e-3，δ_global lr5e-4），1500–3000 步，每步 32–64 张投毒候选；LPIPS 子采样 16 张/步控成本。
- 产物按 siba 约定存：`resource/faat/save_trigger_{nc}_{yt}/`：`global_delta.npy`、`adaptive_delta.npy[N,3,32,32]`（或 π_θ+G 的 state_dict 按需重生成）、`poison_index_map.json`、`opt.log`。

### 4.7 三阶段实现（每阶段定义完成标准）
- **Stage A 规则版 FAAT**（Phase 1，**当前推进中**）：`faat/rules.py` 确定性映射 `s_i→参数`；`δ_global` 复用 Narcissus noise；手写 `Add_Clean_Label_Train_Trigger_faat` + `Add_Test_Trigger_faat`（仅 δ_global）；接入 `train_backdoor.py`。**完成标准**：`--backdoor_type faat` 端到端跑通，ASR 显著高于 Random、L2 ≤ Narcissus 基线。
- **Stage B 可微策略版 FAAT**（Phase 2，核心贡献）：`faat/strategy_net.py`、`trigger_gen.py`、`losses.py`、`optimize.py`、`train_faat.py`；装 SSIM（+可选 LPIPS）。**完成标准**：ASR > Stage A（@1%），BA 与干净基线差 ≤0.5%，且离线 `L_align`（T1）与最终 ASR 在多个 `y_target` 上正相关（验证特征对齐论点），隐蔽性 ≥ Blended/Quantize。
- **Stage C 分层筛选+多样性+可解释**（Phase 3）：`faat/screen.py`（T1 零训练代理评估 / T2 短训 5–10ep / T3 全训 100–200ep+BackdoorBench），`L_div`，π_θ 输出与样本特征的相关性可解释 dump。**完成标准**：候选策略池经 T1→T2→T3 筛选有成本表；最终 FAAT 在 ≥2 个防御上检测率低于 GeneralComponents baseline（matched ASR）。

---

## 五、Agent / 策略模块设计（重点：Agent 必要性的诚实分析）

### 5.1 Agent 必要性结论
- **不需要 RL Agent**（§1.3）。RL 在 4×3090、稀疏反馈、复现性上是净负债。
- **需要「可微摊还策略网络」**：它把逐样本 bi-level 优化摊还为单次前向，是相对 Narcissus 的真贡献。**论文一律用「amortized adaptive policy」，不用「RL Agent」**，避免审稿人攻击必要性。

### 5.2 状态空间（输入 `s_i`，§4.4）
纹理复杂度（Laplacian/Sobel 方差，1）、到 `c_target` 的 L2 与余弦（2）、DCT 8 带能量（8）、选择先验 `res` 类权重（1）。冻结代理上一次性预算，缓存 `[N,d_s]`。

### 5.3 动作空间（输出策略参数）
频带 softmax 权重（B）、通道权重（3，sigmoid×预算）、扰动预算 ε_i（softplus×ε_max）、mask 稀疏 ρ_i（1）、Stage C 的损失权重 λ（4，softmax）。全部可微范围映射。

### 5.4 优化目标（替代 RL 奖励）
端到端可微组合损失 `L`（§4.6）。多目标权衡用**固定权重 + Stage C 的 λ 自整定头**，不做标量奖励。可选地用 **多目标 Pareto 前沿采样**（跑多个 λ 组合，选前沿点）作为增强实验。

### 5.5 训练方式选择（结论）
- **首选：可微端到端优化（amortized）**。✅
- 备选对比（增强实验）：① 上下文 bandit（若想保留「Agent」叙事，作消融）；② 贝叶斯优化（超参级，作敏感性分析）；③ 监督代理（先用逐样本优化造标签再蒸馏，验证摊还质量）。**不做** RL/进化（成本/稳定）。这些备选**只在 Stage C 作为「为何选可微」的对照实验**，不进主线。

### 5.6 如何避免退化为「调参器」
- π_θ 输入是**样本特征** `s_i`，输出随样本变化 → 不是全局标量调参。
- **可解释性证据**（§5.8）必须显示 π_θ 决策与样本纹理/频带**相关**（如高纹理样本系统性地选高频带）。
- 消融：把 π_θ 替换为「全局常数策略」（= 退化为调参器），证明其显著劣化。

### 5.7 如何证明学到了「样本-触发器匹配策略」
- 对抗基线：`π_θ(s_i)` vs `π_θ(shuffle(s_i))`（打乱输入特征）→ 后者 ASR/隐蔽应显著变差。
- 一致性：同类纹理样本的策略输出聚类（t-SNE of `s_i` 着色 by `w_band`）。
- 泛化：在未见过的 `y_target`/类上 π_θ 仍给出合理策略（摊还泛化的证据）。

### 5.8 可解释性实验
- 可视化：逐样本 `w_band` 分布、`δ_adaptive` 范数分布、`||f(x')−c_target||` 直方图（FAAT vs SIBA）。
- 相关性：纹理复杂度 ↔ 选频带；到中心距离 ↔ ε_i。
- 特征空间：FAAT/SIBA/Narcissus 投毒样本在 512d（t-SNE）中相对目标类簇的位置——FAAT 应更「嵌入」目标簇。

---

## 六、实验矩阵（标注【必做】/【增强】/【附录】）

### 6.1 主实验【必做】
FAAT(Stage B) vs {BadNets, Blended, MultiBpp-RGB/B, SIBA, Narcissus, FTROJAN-style}：ASR、BA、隐蔽性(L2/SSIM/GMSD/DCT-L1)、AC/SS 检测率。CIFAR-10 @ {1%, 5%}。

### 6.2 消融【必做】
| 变体 | 去掉什么 | 证明 |
|---|---|---|
| FAAT w/o align | 去 `L_align` | C-机制：特征对齐的价值 |
| FAAT w/o adaptive | π_θ 退化为全局常数（δ_global only） | 自适应 vs 全局 |
| FAAT w/o sample-sel | Random 选择 | 选择联合优化的价值 |
| FAAT w/o δ_adaptive | 只 δ_global | 自适应残差的价值 |
| FAAT fixed-trigger | 用固定触发器替 adaptive | H3 |
| FAAT w/o DCT-band | 不做频带投影 | 频域手段的边际 |

### 6.3 鲁棒性【增强，门控触发】
不同模型（ResNet18/34/50, VGG16, MobileNet——注意 §9.7 仅 ResNet 族有 `extract_feature`）、不同 `y_target`、不同训练轮次、不同防御。**跨数据集见 §6.6 门控**。

### 6.4 可解释性【增强】
§5.8 全套 + π_θ 策略分布 + 触发器与样本特征对齐度量化。

### 6.5 敏感性【增强】
投毒预算、ε、频带数 B、λ 权重、mask 稀疏度的曲线。

### 6.6 跨数据集【附录，强门控】
**仅当 CIFAR-10 上 FAAT 综合超 baseline 后**，按 GTSRB → Tiny-ImageNet → CIFAR-100 顺序扩展（用户硬约束）。每个新数据集需：下载/接入、重算选择指标、重训代理与 `c_target`、重跑主表子集。

### 6.7 防御评估【必做，BackdoorBench】
按 README：保存 `train_backdoor.py` 过滤的投毒索引，替换 BackdoorBench 对应选择逻辑，跑 **Neural Cleanse、STRIP、Spectral Signature、Activation Clustering、Fine-Pruning、ABL、RNP**（SCAn 视 BackdoorBench 支持情况）。报：检测率、误报率、防御后 ASR。

### 6.8 失败案例【必做】
哪些样本 FAAT 翻不转（高 `||f(x')−c_target||` 残差）；哪些类更易失败；π_θ 是否偏向某纹理/频带（过拟合诊断）。

---

## 七、评价指标体系

| 类别 | 指标 | 作用 | 现有/新增 |
|---|---|---|---|
| 攻击有效性 | ASR(PoisonACC) | 主指标 | 现有（parse_detail.py） |
| 正常性能 | BA(CleanACC) | 效用不损 | 现有 |
| 隐蔽-视觉 | L2, L∞ | 扰动幅度 | 已写 metrics/stealth.py |
| 隐蔽-感知 | SSIM, GMSD | 结构/梯度相似 | SSIM 已写 / GMSD 现有 |
| 隐蔽-频谱 | DCT-L1 | 频域可检性 | 已写（用 dct_2d） |
| 隐蔽-深度感知 | LPIPS | 深层感知差 | 新增(Phase2 装 lpips) |
| 语义一致性 | 人工/CLIP-score 辅助 | 触发不破坏语义 | 新增(增强) |
| 防御规避 | TPR@1%FPR, AUC, 防御后 ASR | 对 AC/SS/STRIP/NC/FP/ABL/RNP | 已写 metrics/detection.py(AC/SS) + BackdoorBench |
| 训练成本 | 离线优化时长、GPU·h、π_θ 参数量 | 可行性 | 新增(日志) |
| 稳定性 | 3 种子均值±方差 | 可复现 | 新增(补种子) |
| 可解释 | π_θ-特征相关性、簇内嵌入度 | 策略可解释 | 新增(§5.8) |

---

## 八、论文创新点凝练（机制层，非「用了 Agent/调了参」）

1. **机制层（C-机制）**：提出并验证「干净标签后门中，投毒样本→目标类特征中心的对齐，是同时提升 ASR 与规避聚类防御的充分机制」，用 AC/SS 检测率下降 + ASR 不降为证据。
2. **框架层（C-框架）**：把 GeneralComponents 的「样本适配触发器」推进为「触发器-样本特征几何联合优化」，并以**可微摊还策略网络**把逐样本 bi-level 优化摊还为单次前向（效率/可扩展）。
3. **分析层（C-分析）**：用「样本级 Pareto 异质性」统一解释固定触发器的失败，给自适应触发器的经验收益上界。

**创新性不足时的增强**：① 加 Input-Aware/WaNet 强对照守住 C-框架；② 把 C-机制做成「特征对齐层位消融」（不同 penultimate 层）增强说服力；③ 若特征对齐不显著，转「频带-纹理协同」次要论点。

---

## 九、风险评估与备选方案

### 9.1【研究风险】Agent/摊还效果不明显
- 应对：Stage A 规则版先证 H3；Stage B 必须跑 §5.7 对抗基线（shuffle `s_i`）。若摊还≈全局常数 → 论文降级为「Stage A 规则版 + 机制分析」（§11.1）。

### 9.2【研究风险】与 SIBA/Input-Aware/Narcissus 区分不开
- 应对：主表把 SIBA 作为**头号对照**；消融 §6.2 直接对比「样本特定 vs 特征对齐」。若区分不开 → 强化 C-机制（特征对齐层位消融）或转向效率叙事（摊还 vs Narcissus 逐样本优化的时长对比）。

### 9.3【研究风险】隐蔽性升但 ASR 降
- 应对：C2 决定 ASR 由 δ_global 主导 → 监控 δ_global 的 `L_align`；必要时给 δ_global 单独更大权重/更强对齐损失。降级：放松 `L_perc` 权重换 ASR。

### 9.4【研究风险】防御评估不占优（BackdoorBench 集成坑多）
- 应对：BackdoorBench 集成独立成 Phase 3 末尾的工程任务，预留 1–2 周缓冲；先在仓库内 AC/SS 拿到主结论（不依赖 BackdoorBench）。若 BackdoorBench 跑不通 → 主表用仓库内 AC/SS/FP/STRIP，BackdoorBench 退为附录（与用户「完整集成」决策有冲突时，以此降级保底，并告知用户）。

### 9.5【研究风险】结果不稳定（单种子高方差，复现记录已见）
- 应对：关键结论 3 种子；弱 baseline 不进主表（仅 Random 下界）。报告均值±方差。

### 9.6【降级总案】完整 FAAT 太复杂 → 机制分析型论文
- 退路：以 Phase 0 + Stage A 规则版 + 完整 baseline/防御对比 + 样本级 Pareto 异质性分析，写成「干净标签后门的样本异质性与特征对齐机制分析」论文（§11.1）。合法且有价值。

### 9.7【代码层硬风险（来自源码核查，必读）】
1. **LPIPS/SSIM/kornia 未装** → Phase 0 自写可微 SSIM（已写 `metrics/stealth.py`，不引依赖）；LPIPS 留到 Phase 2 `pip install lpips`，或用代理自身卷积特征做感知替代以零依赖。
2. **测试触发器 index-independent（C2）** → 测试只用 δ_global；优化时 δ_global 必须强对齐。
3. **无代理 checkpoint** → `cal_metric.py` 训了干净 ResNet18 但不存权重（只存 .pkl）。新增独立脚本 `faat/proxy.py` 训并存干净代理权重，**不改 `cal_metric.py`**（避免污染上游指标管线）。
4. **`extract_feature` 仅 ResNet 族有**（`cifar_resnet.py`）→ FAAT 的 `L_align` 限定 ResNet18/34/50；其他架构在主表外或加适配层。文档化为已知限制。
5. **增强在注入后（C3）** → δ_global 分散、δ_adaptive 限低频带，抗 RandomCrop/Flip。
6. **DCT 输入范围** → `low_freq`（`utils.py:89`）假设 `[-1,1]`，别照抄；FAAT 全程 `[0,1]`，`dct_2d` 与尺度无关。
7. **`Add_Clean_Label_Train_Trigger_*` 重标为目标类** → FAAT 沿用以保可比（§0 澄清）。
8. **`parse_detail.py`/`parse_results.py` 依赖 ASR/BA 列位置** → 新增隐蔽/检测度量另起 JSON 合并，**不挪动** ASR/BA 列。
9. **训练循环原不保存模型权重** → 已在 `train_backdoor.py` 末轮增加 `model_last.pth`/`args.json`/`poison_inds.json` 保存（additive，try/except）。
10. **SIBA 路径错配** → 代码读 `./resource/save_trigger_10_0/`，实际在 `./resource/siba/save_trigger_10_0/{uap,mask}.npy`。跑 SIBA 前需 symlink 或加 `--save_trigger` 覆盖。

---

## 十、阶段计划与时间安排（标准论文版，10–12 周，CIFAR-10 为主锚）

> 门控：Week 1–8 全在 CIFAR-10；Week 9 的「超 baseline」判定通过后，Week 10–12 才启动跨数据集（且按序）。每周给出【本周目标/实验/产出/进入下周的标准/失败调整】。
> 现实约束：4 GPU 与他人共享；baseline 复现由 `scripts/run_table1.sh` 承担，FAAT 专项训练排在其间隙/空闲卡上。

**Week 1 — Phase 0 基建 + 度量补齐【必做，部分已完成】**
- 目标：复现管线产出 baseline + 补 SSIM/DCT-L1/AC/SS 度量 + 增加模型 checkpoint 保存。
- 实验：`scripts/run_table1.sh` 跑 Badnets-C/MultiBpp-B（+ Blended-C 定夺后）；写可微 SSIM、DCT-L1、AC、SS（✅ 已写）；`train_backdoor.py` 末轮存权重（✅ 已加）。
- 产出：`results/<attack>_<sel>/` × {ASR,BA,L2,SSIM,GMSD,DCT-L1,AC,SS} 表；强度梯度 Pareto 图。
- 进下周：baseline 跑通 + 4 度量可复现 + checkpoint 可加载。
- 失败调整：度量大错 → 先用 GMSD（已有）兜底，SSIM/AC/SS 排查。

**Week 2 — Phase 0 假设验证【必做】**
- 目标：判定 H0/H1/H2/H3。
- 实验：强度梯度 Pareto（H0）；逐样本翻转-隐蔽-特征距离散点（H1）；规则版自适应 vs 全局（H3）；初步「拉特征到中心」消融（H2）。
- 产出：H0–H3 判定报告 + go/no-go（写入 `docs/PHASE0-REPORT.md`）。
- 进下周：H0∧H1 成立 ∧ H3 至少持平全局。
- 失败调整：H0/H1 不成立 → 触发 §9.6 降级，与用户确认转分析型论文。

**Week 3 — Stage A 规则版 FAAT【必做，当前推进中】**
- 目标：`--backdoor_type faat` 端到端跑通。
- 实验：`faat/rules.py`、`Add_Clean_Label_Train_Trigger_faat`/`Add_Test_Trigger_faat`（仅 δ_global=Narcissus noise）、`train_backdoor.py` 分支、`faat/proxy.py`（干净代理+c_target）。
- 产出：规则版 FAAT 的 ASR/BA/隐蔽/检测 行；证明 ASR > Random、L2 ≤ Narcissus。
- 进下周：端到端通 + 非平凡 ASR。
- 失败调整：注入接口错 → 对齐 `Add_Clean_Label_Train_Trigger_NAR`（`utils.py:422`）模板。

**Week 4 — Stage B 核心：策略网络 + 生成器 + 损失【必做】**
- 目标：可微 FAAT 离线优化跑通。
- 实验：`strategy_net.py`、`trigger_gen.py`（复用 `UnetGenerator`）、`losses.py`（L_align/L_freq/L_perc）、`optimize.py`、`train_faat.py`；装 lpips。
- 产出：离线 `opt.log`（L_align 等曲线）+ artifact；与 Stage A 对比。
- 进下周：ASR > Stage A。
- 失败调整：δ_global 被平均化不学 → 给 δ_global 独立更大 lr/对齐权重（C2）。

**Week 5 — Stage B 验证 + 主实验初版【必做】**
- 目标：验证摊还论点 + 主表初版。
- 实验：§5.7 shuffle 对抗基线；FAAT vs 全 baseline @1%/5%。
- 产出：主表初版 + 摊还有效性证据。
- 进下周：FAAT CIFAR-10 主表 ≥ baseline。
- 失败调整：ASR 不足 → §9.3 放松隐蔽权重。

**Week 6 — 消融 + 敏感性【必做】**
- 实验：§6.2 全消融；§6.5 敏感性曲线。
- 产出：消融表 + 敏感性图。
- 进下周：消融支持 C-机制/C-框架。
- 失败调整：消融不支持 → 强化 §8 增强项。

**Week 7 — 仓库内轻量防御 + 可解释【必做】**
- 实验：AC/SS/FP/STRIP 仓库内实现 + §5.8 可解释 dump。
- 产出：防御表（仓库内）+ 可解释图。
- 进下周：FAAT 在 ≥2 防御上检测率低于 baseline。
- 失败调整：防御不占优 → §9.2 转效率叙事。

**Week 8 — 多种子稳定性 + 失败案例【必做】**
- 实验：主表 3 种子；§6.8 失败案例。
- 产出：均值±方差主表 + 失败分析。
- 进下周：结论稳定。

**Week 9 — 【门控判定】CIFAR-10 是否综合超 baseline【必做】**
- 判定：ASR/BA/隐蔽/检测 综合是否超 {MultiBpp-RGB/B, SIBA, Narcissus}。
- 通过 → Week 10 启动 BackdoorBench 集成 + 跨数据集；未通过 → 集中修 FAAT 或触发降级。
- 产出：门控判定报告 `docs/GATE-CIFAR10.md`（用户复核）。

**Week 10 — BackdoorBench 集成【必做，工程】**
- 实验：按 README 接入索引，跑 NC/STRIP/SS/AC/FP/ABL/RNP。
- 产出：BackdoorBench 防御表。
- 失败调整：§9.4，仓库内防御兜底，BackdoorBench 退附录。

**Week 11 — 跨数据集（GTSRB→Tiny-ImageNet→CIFAR-100，按序）【附录，门控】**
- 仅 Week 9 通过才做；每数据集主表子集（下载/选择指标/代理重算）。
- 产出：跨数据集表（视时间覆盖 1–3 个）。

**Week 12 — 论文组织 + 图表定稿【必做】**
- 产出：论文初稿 + 全部图表 + 消融/防御/可解释附录。
- 失败调整：未完成项移入 Future Work。

---

## 十一、最终输出三层方案

### 11.1 最小可发表版本（保底，6–8 周可达）
Phase 0 + Stage A 规则版 FAAT + {BadNets,Blended,MultiBpp-RGB/B,SIBA,Narcissus} + 仓库内 AC/SS/FP + 样本级 Pareto 异质性分析。定位「干净标签后门的样本异质性与特征对齐机制分析」。**当 Stage B 受阻或创新性被证伪时启用。**

### 11.2 标准论文版（主目标，10–12 周）
完整 FAAT（Stage A→B）+ 全 baseline + §6.2 全消融 + §6.5 敏感性 + 仓库内防御 + BackdoorBench（Week 10）+ 3 种子稳定性 + 失败案例。CIFAR-10 为主战场。

### 11.3 强化版（12+ 周，视 Week 9 门控与时间）
标准版 + Stage C 分层筛选/`L_div`/λ 自整定 + §5 摊还 vs bandit/BO 对照 + §5.8 深度可解释 + 跨数据集（GTSRB/Tiny-ImageNet/CIFAR-100 全覆盖）。赌 C-机制+C-框架双贡献。

---

## 附录 A：文件级实现清单（给 Claude Code 执行用，源自源码核查）

**新增文件**（`GeneralComponents-main/faat/`）：
- `__init__.py`、`rules.py`(Stage A)、`strategy_net.py`(`FeatureSummary`+`StrategyNet`)、`trigger_gen.py`(`FAATGenerator`，复用 `UnetGenerator`)、`losses.py`(L_align/L_freq/L_perc/L_div + 自写 SSIM)、`proxy.py`(干净代理 + `c_target`)、`optimize.py`(离线优化循环)、`apply_trigger.py`(`Add_Clean_Label_Train_Trigger_faat` / `Add_Test_Trigger_faat`)、`screen.py`(T1/T2/T3)。
- 顶层 `train_faat.py`：优化→存 artifact→（可选）转交 `train_backdoor.py`。
- Phase 0 度量（✅ 已建）：`metrics/stealth.py`(SSIM/DCT-L1/L2/Linf)、`metrics/detection.py`(AC/SS)、`metrics/smoke_test.py`。

**改动现有文件**：
- `train_backdoor.py`：`--backdoor_type` 加 `'faat'`(L87)；L241-260 加 faat 分支（仿 narcissus/siba）；日志不挪 ASR/BA 列；末轮存模型权重（✅ 已加）。
- `parse_results.py` / `parse_detail.py`：向后兼容地加可选隐蔽/检测度量合并（`--stealth <json>` / `--detect <json>`），不动 ASR/BA 正则。
- 不改：`cifar_resnet.py`/`dct.py`/`cal_metric.py`/`models/*`（复用）。

**关键复用点**：注入模板 `Add_Clean_Label_Train_Trigger_NAR`(`utils.py:422`)；测试模板 `Add_Test_Trigger_NAR`(`utils.py:40`)；`UnetGenerator`(`cifar_resnet.py:11`)；`extract_feature`(`cifar_resnet.py:160`，512d)；`dct_2d/idct_2d`(`dct.py:85/99`)；`get_stats`+`resource/save_metric_10_res`(选择指标)；Narcissus noise `resource/narcissus/noise_01000.pth`。

---

## 附录 B：实验记录规范（给 Claude Code 记录用，与 result_all.md 一致）

**目录**：`results/<attack>_<selection>[_<res_sel>][_pr<poison_rate>_yt<target>]/seed<seed>/`（遵循 `result_all.md` 命名），内含：
- `output_<seed>.log`（沿用现有训练日志格式，ASR/BA 列不变）。
- `args.json`（完整 argparse 快照，✅ 已自动保存）。
- `metrics.json`：`{ASR, BA, L2, Linf, SSIM, GMSD, dct_l1, lpips?, AC:{tpr@1,auc}, SS:{tpr@1,auc}, defense_post_ASR?, opt_time_s, gpu_h, params}`。
- `model_last.pth`（末轮权重，✅ 已自动保存，供 AC/SS/特征分析）。
- `poison_inds.json`（投毒索引，✅ 已自动保存）。
- `artifacts/`：触发器（global_delta.npy/adaptive_delta.npy 或 state_dict）、poison_index_map.json。
- `figs/`：强度-Pareto、逐样本散点、t-SNE、策略分布等 PNG + 源数据 csv。

**规则**：每个 run 必须可由 `args.json` + seed 复现；隐蔽/检测度量对**所有**方法在同一投毒集计算；3 种子的关键 run 命名加 `_seed{1,2,3}` 并聚合到 `summary.json`。**每跑完一组必跑 `parse_detail.py --write-md docs/result_all.md`**（CLAUDE.md 强制）。Phase 0 产出统一汇总到 `docs/PHASE0-REPORT.md`，Week 9 门控结论写入 `docs/GATE-CIFAR10.md`（用户复核）。

---

## 验证（如何端到端检验本计划可执行）
1. **Phase 0 烟测（✅ 已通过）**：2-epoch quantize 训练产出 `model_last.pth`+`args.json`+`poison_inds.json`；`metrics/smoke_test.py` 在其上跑通 stealth(L2/Linf/SSIM/DCT-L1) + detection(AC/SS AUC/TPR@1%FPR)。
2. **复现管线烟测**：`CUDA_VISIBLE_DEVICES=<空闲卡> python train_backdoor.py --backdoor_type siba --selection res --res_sel square --poison_rate 0.01 --y_target 0 --select_epoch 10 --output_dir ./resource/save_metric_10_res --result_dir results/phase0_siba_smoke --epochs 2`（先修 SIBA 路径）能跑出 ASR/BA。
3. **Stage A 烟测**：`--backdoor_type faat`（规则版）端到端产出 ASR>Random、L2≤Narcissus。
4. **Stage B 论点检验**：多个 `y_target` 下离线 `L_align` 与最终 ASR 正相关（特征对齐论点的直接证据）。
5. **门控**：Week 9 `docs/GATE-CIFAR10.md` 给出 CIFAR-10 是否综合超 baseline 的明确判定，决定是否启动跨数据集与 BackdoorBench 全量。
