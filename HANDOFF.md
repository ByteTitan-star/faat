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

---

## 2. 战略决策演进（作者意图史，勿回退）

1. **09-11**：评估 work7-7month 可发表性 → FAAT 论文成熟但 novelty 承压。
2. **09-11**：作者选定了当时的新主线"OOD-Calibrated Target-Specific Trigger"（OOD 做显式负校准参照，区别于 Narcissus 的 surrogate 用法）；定 48h 验证纪律 + 预注册 Go/No-Go。
3. **09-12**：48h 结果 **NO-GO**——OOD 负校准作为攻击组件被证伪（GTSRB c 臂 ASR 3.3/4.0 vs a 臂 49.5/50.7）。作者当日决定**转向防御导向表征研究**（可学性边界 + 检测可观测性），OOD 降为实验变量。
4. **09-12**：作者做了第二轮 novelty 核验（15 条修正），定稿表述安全区（见 §4）。
5. **09-16**：训练收官后，作者问"触发器强还是防御强"，并做最终战略决策：**分开两篇**。理由：叙事互斥（攻击 vs 防御表征）、时间线不同步（FAAT 能投、表征还要补）、风险不隔离（FAAT novelty 承压不应拖累表征线）、实验设定不同源（表征用简化触发器族，非完整 FAAT）。
6. **可选的桥（已同意的方向）**：把 FAAT 作为相图中的一个额外触发器族跑进边界图（堵"只扫了一个攻击族"的审稿问题）——这是扩展不是合并。

**三篇论文的叙事弧**：Wicked Oddities（ICLR 2025，投毒哪些样本）→ FAAT（用什么触发器）→ 表征论文（什么时候学得会、什么时候检得到）。

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

### 线 A（FAAT，1-2 周内可投）
1. 补 `paper/README.md` 列出的 `—` 占位（T1 4 格、T4 5 格、supp 2 格）；
2. **核实 ablation_CLEAN 对照疑点**（ASR 61.7/90.3 疑非纯 clean，`results/ablation_CLEAN_*`）；
3. related work 加 NoiseAttack 区分（KST 部分）；refs.bib verify 标记清查；
4. 加 Limitations 节；决定会议（CVPR 类攻击侧）。

### 线 B（表征，补缺口→写作）
1. **cross-architecture 验证**（最优先——单架构是最大 caveat）：`cifar_resnet.py` 有 VGG 等现成变体，重跑 a/c/cur 臂即可；
2. 检测层补全：STRIP（`faat/defenses.py:strip_detection`）、NC（未实现，需写）、FP（`defenses.py:fine_pruning_defense`）；
3. 可选：FAAT-as-trigger-family 进相图（用冻结仓库配方）；
4. 写作：相图主图（预算×校准×P_L）+ 机制链图 + 检测解耦散点；会议定位 S&P/USENIX/CCS analysis 风格或 ICLR/NeurIPS。

---

## 7. 基础设施索引（线 B）

- 启动：`scripts/run_ood_abc.sh <cifar10|cifar100|gtsrb> <gpu> <seed> <a|b|c|cur...>`，env：`EPOCHS/EPOCH_TAG/POISON_RATE/PRTAG/OOD_WEIGHT/WTAG`
- 四臂语义：a=target-only 正对齐 / b=OOD 混入 pool / c=OOD 负校准（weight 可调）/ cur=非目标类池（同管线锚点）
- 监控：`scripts/watch_ood_abc.sh`（51 tags，自动 STATUS.md + hd0 备份）；看板 `results_ood/STATUS.md`
- 分析：`scripts/pilot_metrics.py`（行为+proxy 通用度）→ `docs/pilot_metrics.csv`；`scripts/pilot_representation_ext.py`（gini/CKA/增广）→ `docs/pilot_representation.csv`；`scripts/analyze_all.py`（五表汇总）；`scripts/pilot_figures.py`（图）
- 汇总文档：`docs/PILOT-OBSERVATIONS.md`（Obs1-6）、`docs/ood_abc_results.md`（NO-GO 判定史）、`CLAUDE.md`（纪律全文）

---

## 8. 已知坑（重踩过，勿再踩）

1. train_backdoor 日志按 seed 命名：seed2/3 是 `output_2/3.log`，不是 `output_1.log`；
2. 多进程**不得**追加写同一日志文件（互覆写，丢输出）；
3. Bash 后台链 `cd X && nohup ... & cmd2` 的 `&` 会把 cd 困在子壳——统一用 `setsid nohup bash -c "cd ABS && ..."` + 绝对路径日志；
4. GPU 与他人共享：只占 ≥4GB 空闲的卡；瞬时 CUDA fault（unspecified launch failure）发生过 3 次，失败 run 查启动器日志后重发即可；
5. `results_ood/` 被 gitignore（备份走 tarball+分支约定）；`pkill -f` 会匹配到自己的包装 shell，用 `^bash scripts/...` 锚定；
6. conda env 无 activate 脚本，直接用 `/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python`；
7. 旧仓库 `GeneralComponents-main` 已冻结，只读引用，不写回。
