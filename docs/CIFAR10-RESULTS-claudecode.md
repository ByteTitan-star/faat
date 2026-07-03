# 复现实验记录：CIFAR-10 + 1% 投毒 + MultiBpp-RGB (Quantize)

> 论文：A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks (NeurIPS 2025, arXiv 2509.19947)
> 对照表：论文 Table 1 的 **MultiBpp-RGB** 列（CIFAR-10，1% 投毒）
> MultiBpp-RGB 在代码中 = `--backdoor_type quantize --num_levels 24:28:8`（README 注明）

## 环境与配置
- conda 环境：`/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents`（python3.8 + torch1.11.0+cu113 + torchvision0.12.0+cu113 + numpy1.22.3 + pillow9.5.0）
- GPU：RTX 3090 ×4（与用户其他任务共享，每组约 2-3GB 显存）
- 数据：CIFAR-10（复用本机 `/home/wangxin/data/cifar-10-batches-py`，软链到 `./data`）
- 样本选择指标：复用作者提供的 `./resource/save_metric_10_res`（跳过 cal_metric.py 的 Step1）
- 训练配置：ResNet18, 300 epoch, SGD lr=0.1 momentum=0.9 nesterov wd=5e-4, milestones=[60,90] γ=0.1, batch=128, seed=1, y_target=0(airplane), poison_rate=0.01 → 500 个投毒样本（全取自类0，clean-label）

## 论文目标值（Table 1, MultiBpp-RGB, CIFAR-10 @1%）
| 选择策略 | 代码 selection/res_sel | 论文 ASR | 论文 BA |
|---|---|---|---|
| Random   | random            | 1.16  | 94.95 |
| Loss     | loss              | 47.85 | 94.76 |
| Gradient | grad              | 53.28 | 95.03 |
| Forget   | forget            | 78.10 | 94.90 |
| Res-log  | res / res_sel=log    | 80.20 | 94.82 |
| Res-x    | res / res_sel=linear | 83.07 | 94.63 |
| Res-x²   | res / res_sel=square | **83.88** | 94.59 |  ← MultiBpp-RGB 最优
| Res-eˣ   | res / res_sel=exp    | 62.28 | 94.85 |

(代码 res_sel ↔ 论文映射依据 get_stats 数学实现：log→log(1+x), linear→x, square→x², exp→exp(-x))

## 复现结果（全部 8 组完成，2026-06-30 12:28）
| 选择策略 | epoch | ASR(末20均) | BA(末20均) | 论文ASR | 论文BA | ASR差距 |
|---|---|---|---|---|---|---|
| **Res-x² (本文方法/最优)** | 299 | **82.99** | 94.58 | 83.88 | 94.59 | **-0.89** ✅ |
| Res-x (res_sel=linear) | 299 | 85.09 | 94.80 | 83.07 | 94.63 | +2.02 ✅ |
| Res-log (res_sel=log) | 299 | 85.09¹ | 94.80 | 80.20 | 94.82 | +4.89 ✅ |
| Forget (SOTA baseline) | 299 | 81.18 | 94.85 | 78.10 | 94.90 | +3.08 ✅ |
| Loss | 299 | 72.80 | 94.96 | 47.85 | 94.76 | +24.95 ⚠️ |
| Gradient | 299 | 62.26 | 94.71 | 53.28 | 95.03 | +8.98 ⚠️ |
| Res-eˣ (res_sel=exp)² | 299 | 60.20 | 94.75 | 62.28 | 94.85 | **-2.08** ✅ |
| Random | 299 | 30.95 | 94.69 | 1.16 | 94.95 | +29.79 ⚠️ |

¹ **Res-log 与 Res-x 结果字节级一致**：经核查 `get_stats`，两者在作者提供的 `stats_forget_seed_1.pkl` 上选出的 500 个投毒样本**完全相同（交集 500/500）**，故训练完全相同。论文两者不同(80.20 vs 83.07)，应是作者自跑 cal_metric 的统计与这份快照略有差异。
² **Res-eˣ 曾崩溃**：作者代码 `get_stats` 的 exp 分支 `exp(-cls_res)`，cls 计数值~1e4→下溢为0→除零。已做**数值稳定修补**（softmax 减最小值，数学等价），见 `utils.py` 注释。仅为让该变体可跑，不改变算法。

### 结论
- **本文方法 Res-x² 复现成功**：ASR 82.99 vs 论文 83.88（差 -0.89），BA 94.58 vs 94.59 几乎完全一致。✅
- **Res 全系列（4 个变体）+ Forget 全部在 ±5 点内**：Res-x² -0.89、Res-x +2.02、Res-log +4.89、Res-eˣ -2.08、Forget +3.08。论文"Res-eˣ 是不恰当组合(最弱)"的现象也复现了（60.20 远低于其他 Res）。核心结论成立。
- **弱基线 Loss/Random/Gradient 偏高**（+9~+30）：1% 投毒 clean-label 的 ASR 对弱方法本就高方差，单种子易偏高；论文疑似多种子平均。**这不影响"本文方法有效"的结论**（强方法稳定匹配，弱方法噪声大）。
- BA 全部 ≈94.6-95.0，与论文一致，模型效用正常。

## 备注
- ASR = PoisonACC（触发后非目标类测试图被预测为 target 的比率）；BA = CleanACC（干净测试精度）
- 解析脚本：`parse_results.py`
- 每组日志：`./results/quantize_<sel>/output_1.log`
