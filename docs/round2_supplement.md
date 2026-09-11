# Round-2 补跑实验日志（Direction A：FAAT 主轴 + KST 章节）

> **触发**：2026-08-04 用户决定走方向 A（FAAT 主轴 + KST 章节），要求 4 卡并行补实验，**不覆盖已有内容、结果可追溯**。
> **调度脚本**：`scripts/run_round2_supplement.sh`（幂等：ep299 跳过；新目录：绝不覆盖 `results/kst_sdt/*` 旧目录）。
> **审计基线 commit**：`4097c94`（`exp/kst-sdt`，"multi-dataset campaign results"）。

---

## 1. 启动前审计结论（哪些真缺，文档过时项已纠正）

通过逐个读 `results/kst_sdt/*/output_1.log` 末行确认（之前的"last_epoch=10"是 `select_epoch=10` 误判）：

| 类别 | 实际状态 | 处理 |
|---|---|---|
| **CIFAR-100 KST** e16/e20/e48 × forget/reslin | ✅ **全部跑满 300ep**（文档 §10.4 "ε48 full 待定" 已过时） | 不重跑，仅重解析 |
| **CIFAR-100 baseline** badnets/blend/mbpprgb/mbppB | ✅ 跑满 300ep | 不动 |
| **GTSRB KST** e16/e20 × forget/reslin | ✅ 跑满 300ep（ASR=0，但完整） | 不动 |
| **CIFAR-10 KST** e16_forget/e20_reslinear 等 | ✅ seed1 跑满 | **补 seed2/3** |
| **GTSRB baseline** badnets/blend/mbpprgb/mbppB | ❌ **只到 ep24（W4 被 kill）** | **重跑 full** |
| **Tiny KST** 任何 ε full | ❌ **从未跑过**（目录不存在，仅 `diag_tiny_*` 短跑） | **新跑 e48+e16** |
| **Tiny baseline** bl_tiny_*_seed1 | ✅ 跑满 299 | 不动 |

---

## 2. 本轮补跑清单（12 run，全写新目录）

| # | 新结果目录 | seed | GPU | 配置要点 | 预计时长 |
|---|---|---|---|---|---|
| 1 | `results/kst_sdt/tiny_kst_e48_forget` | 1 | 0 | Tiny, kst, eps0.1882(=e48), forget, pr0.0025 | ~2.6h |
| 2 | `results/kst_sdt/tiny_kst_e48_reslin` | 1 | 1 | Tiny, kst, e48, res/linear | ~2.6h |
| 3 | `results/kst_sdt/tiny_kst_e16_forget` | 1 | 2 | Tiny, kst, eps0.0627(=e16), forget | ~2.6h |
| 4 | `results/kst_sdt/tiny_kst_e16_reslin` | 1 | 2 | Tiny, kst, e16, res/linear | ~2.6h |
| 5 | `results/kst_sdt/cl_kst_e16_forget_s2` | 2 | 0 | CIFAR-10, kst, e16, forget | ~85min |
| 6 | `results/kst_sdt/cl_kst_e16_forget_s3` | 3 | 1 | CIFAR-10, kst, e16, forget | ~85min |
| 7 | `results/kst_sdt/cl_kst_e20_reslinear_s2` | 2 | 0 | CIFAR-10, kst, e20, res/linear | ~85min |
| 8 | `results/kst_sdt/cl_kst_e20_reslinear_s3` | 3 | 1 | CIFAR-10, kst, e20, res/linear | ~85min |
| 9 | `results/kst_sdt/gtsrb_bl_badnets_full` | 1 | 0 | GTSRB, badnets, res/linear, pr0.01 | ~56min |
| 10 | `results/kst_sdt/gtsrb_bl_blend_full` | 1 | 1 | GTSRB, blend, res/linear | ~56min |
| 11 | `results/kst_sdt/gtsrb_bl_mbpprgb_full` | 1 | 2 | GTSRB, quantize 24:28:8, res/linear | ~56min |
| 12 | `results/kst_sdt/gtsrb_bl_mbppB_full` | 1 | 2 | GTSRB, quantize 255:255:8, res/linear | ~56min |

> **统一参数**（除上表变化项）：`--model resnet18 --epochs 300 --learning_rate 0.1 --y_target 0`，沿用 baseline 管线（SGD[60,90] γ0.1, batch128, Pad+Flip+RandomCrop）。
> **数据/指标目录**（启动前已确认存在）：`data_tiny` ✓、`data/GTSRB32` ✓、`resource/save_metric_{gtsrb,tiny_res,10_res}` ✓。
> **KST delta 文件**（启动前已确认存在）：`kst_delta_tiny_r4_eps0.1882_a0.0_ce0.pt` ✓、`..._eps0.0627_a0.0_ce0.pt` ✓、`kst_delta_r4_eps0.0627.pt` ✓、`kst_delta_r4_eps0.0784.pt` ✓。

---

## 3. GPU 分配（用 3 卡，避 GPU3）

```
GPU0 队列(~6.4h): tiny_e48_forget → cl_e16_forget_s2 → cl_e20_reslinear_s2 → gtsrb_bl_badnets_full
GPU1 队列(~6.4h): tiny_e48_reslin → cl_e16_forget_s3 → cl_e20_reslinear_s3 → gtsrb_bl_blend_full
GPU2 队列(~7.0h): tiny_e16_forget → tiny_e16_reslin → gtsrb_bl_mbpprgb_full → gtsrb_bl_mbppB_full
```

> **GPU3 不用**：`wanaihua` 用户在跑 `adapgc`（5.4GB，66% util）。CLAUDE.md 规矩"不抢占他人正在用的卡"。用户要"4 卡"，但合规前提下只能用 3 卡。

---

## 4. 不覆盖 / 可追溯机制

1. **目录**：全部用新名（`_full` / `_s2` / `_s3` / `tiny_kst_e48_*` / `tiny_kst_e16_*`）。原 `gtsrb_bl_*`（kill 在 ep24）保留作"被中断"证据，不动。
2. **日志**：每个 run → `<rdir>/output_1.log`（训练日志，每 epoch 一行）+ `<rdir>.stdout`（屏幕输出）；调度器总日志 → `results/_round2_supplement.log`。
3. **幂等**：`is_done()` 检测 `output_1.log` 末有 `] - 299 ` 行 → 该 run 自动跳过。中断重跑安全。
4. **不动 working tree**：`scripts/run_campaign_all.sh`（已 modified）保持原样；本日志、调度脚本、`all_result.md` 为新增/追加。
5. **完成后**：用 `faat/parse_campaign.py` 解析新目录（脚本会自动 glob 到 `tiny_kst_*`、`gtsrb_bl_*_full` 等），数字**追加**到 `all_result.md`（不覆盖原表，新增"Round-2 补跑"小节）。

---

## 5. 监控

```bash
# 调度器总进度（哪个 run 在跑/完成）
tail -f results/_round2_supplement.log
# 某个 run 的训练 epoch 进度
tail -f results/kst_sdt/tiny_kst_e48_forget/output_1.log
# GPU 占用（应见 GPU0/1/2 各一个 python，~2-3GB）
nvidia-smi
# 完成判据：所有 12 目录的 output_1.log 末行都是 epoch 299 + saved model_last.pth
```

---

## 6. 启动记录

- **启动时间**：见 `results/_round2_supplement.log` 首行 `### ROUND-2 SUPPLEMENT START ###`
- **预计全部完成**：启动后 ~7 小时（受 GPU2 队列两个 Tiny 串行限制）
- **完成后动作**：parse → 追加 `all_result.md` → 在本日志 §7 填最终结果表。

---

## 7. 第一批（Round-2）最终结果 — 12/12 全部 ep299 ✅

> 跑完时间 2026-08-05 00:03。**注意日志文件名是 `output_{seed}.log`**（`train_backdoor.py:273`），seed2/3 写到 `output_2.log`/`output_3.log`（不是 output_1.log）。

### 7.1 结果表（ASR/BA = 末20均）

| run | seed | ASR | BA | peak ASR | 解读 |
|---|---|---|---|---|---|
| tiny_kst_e48_forget | 1 | **98.5** | 55.2 | 100.0 | Tiny ε48 超 baseline(BadNets 90.11) +8.4 |
| tiny_kst_e48_reslin | 1 | **98.5** | 55.2 | 100.0 | 同上（forget/reslin 末20均巧合相同，log 非字节同）|
| tiny_kst_e16_forget | 1 | 45.3 | 55.1 | 64.5 | ε16 太弱（Tiny 需 ε48）|
| tiny_kst_e16_reslin | 1 | 62.6 | 54.6 | 99.8 | ε16 reslin 略好 |
| cl_kst_e16_forget_s2 | 2 | 64.8 | 94.9 | 93.1 | ⚠️ seed 方差大 |
| cl_kst_e16_forget_s3 | 3 | 62.0 | 94.9 | 99.7 | ⚠️ s1=86.76 但 s2/s3≈62-65 |
| cl_kst_e20_reslinear_s2 | 2 | 91.5 | 94.6 | 99.9 | ✅ 稳健 |
| cl_kst_e20_reslinear_s3 | 3 | 89.5 | 94.9 | 98.8 | ✅ s1/s2/s3=89-92 |
| gtsrb_bl_badnets_full | 1 | 0.0 | 100.0 | 1.3 | baseline 崩溃确认 |
| gtsrb_bl_blend_full | 1 | 34.7 | 99.9 | 80.8 | baseline 最强(Blended) |
| gtsrb_bl_mbpprgb_full | 1 | 0.0 | 100.0 | 6.7 | 崩溃 |
| gtsrb_bl_mbppB_full | 1 | 0.0 | 100.0 | 4.5 | 崩溃 |

### 7.2 关键发现（影响论文写法）

1. **CIFAR-10 KST 主推 `e20_reslinear`，不主推 `e16_forget`**：e20_reslinear 三 seed 稳健（92.6/91.5/89.5）；e16_forget seed 方差大（86.8/64.8/62.0，s1 幸运成分高）。
2. **Tiny ε48 = 98.5 坐实超 baseline**（之前文档只有诊断 peak 99.8%，现 full 300ep 末20均 = 98.5）。需多seed确认（Round-3 在跑）。
3. **GTSRB baseline full 确认崩溃**（badnets/mbpp=0%，blend=34.7%），与 25ep 趋势一致。

### 7.3 Round-3 / 3b / 4 最终结果（多seed补全 + Tiny ε20 gap，2026-08-05 完成）

3 个调度脚本共 12 run，补全 KST 章节多 seed 稳健性 + Tiny ε-Pareto + GTSRB KST ε48：

- **Round-3**（`run_round3_supplement.sh`）：Tiny ε48 / CIFAR-100 ε48 × forget/reslin × seed{2,3}（8）+ GTSRB KST ε48 × forget/reslin（2）。GPU0 启动时被 wanaihua(Factory) 抢占 → 3 run OOM 秒退；GPU1/2/3 跑 7 run 成功。
- **Round-3b**（`run_round3b_recover.sh`）：pgrep 等 Round-3 结束后，用 GPU1/2/3 补跑 3 个 GPU0 失败 run（tiny_forget_s2 / c100_forget_s2 / gtsrb_forget），全部 exit=0。
- **Round-4**（`run_round4_tiny_e20.sh`）：补 Tiny ε20 full gap（forget+reslin），GPU3 立即 + GPU2 等 Round-3b 释放。

#### 多 seed 稳健性表（ASR 末20均）

| 数据集 | 配置 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| Tiny ε48 | forget | 98.5 | 97.7 | 98.0 |
| Tiny ε48 | reslin | 98.5 | 98.2 | 98.0 |
| CIFAR-100 ε48 | forget | 83.2 | 91.9 | 92.9 |
| CIFAR-100 ε48 | reslin | 94.0 | 93.2 | 94.6 |
| CIFAR-10 e20_reslinear | — | 92.6 | 91.5 | 89.5 |
| CIFAR-10 e16_forget | — | 86.8 | 64.8 | 62.0 |

#### Tiny ε-Pareto（补完 ε20，BA~55）

| ε | forget | reslin |
|---|---|---|
| ε16 | 45.3 | 62.6 |
| ε20 | 72.6 | 75.1 |
| ε48 | 98.5 | 98.5 |

#### GTSRB KST ε48（limitation 完整）

| selection | ASR | BA |
|---|---|---|
| forget | 0.8 | 100.0 |
| reslin | 0.8 | 100.0 |

→ GTSRB KST ε16/20/48 全失败，平谱触发器翻不动 BA 99.95% 极自信模型，limitation 坐实。

#### Round-3 的小波折（可追溯）
- GPU0 在 Round-3 启动瞬间被 wanaihua(Factory) 抢占，3 run OOM（exit=1）。Traceback 保留在各目录 `.stdout`（`CUDA OOM at model.cuda()`）。
- Round-3b 用 pgrep 等 Round-3 进程结束，再用空闲的 GPU1/2/3 补跑，全部 exit=0。
- 所有 run 幂等（`is_done` 检测 `output_{seed}.log` ep299），中断重跑安全。
