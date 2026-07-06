# CLAUDE.md — 项目指南（Claude Code 每次会话自动加载）

> ## ✅ Table 1 复现完成 32/32（2026-07-01 21:04 全部跑满 epoch=299）—— 已暂停，等 FAAT agent 改完代码
> 全 32 组完成，调度器已自动 `parse_detail.py --write-md` 生成 §3.1（32 行）。**代码完整性已核**：`utils.py`(blend 逻辑+`get_stats`)/`cifar_resnet.py`(网络) 全程未动，`train_backdoor.py` 仅被 FAAT 加了 `faatb` 分支（纯加法，blend 路径 `diff` 无变化），故 32 组结果**有效、未被污染**。
> **现在暂停**：等 FAAT agent 改完代码、结果稳定后，再决定是否重新复现/跑后续表（CIFAR-100/消融/防御，见 `docs/result_all.md` §1）。恢复前若担心污染可抽查 1 组重跑比对。
> **不要重启 `scripts/run_table1.sh`、不要从头重跑 32 组**：已完成且有效。完整状态见 `docs/result_all.md`。

本文件给 Claude Code 的**持久化工作指南**。开始任何复现/训练任务前先读这里。

## Git 仓库与分支管理（强制，防版本代码/数据丢失）

> 远程：`git@github.com:ByteTitan-star/faat.git`（SSH）或 `https://github.com/ByteTitan-star/faat.git`（HTTPS）。
> 凭证已缓存 `~/.git-credentials`（chmod 600）。当前服务器 GitHub 网络不通（代理超时），恢复后用 `git push --all origin && git push --tags` 推送。
> 本地快照双保险：`snapshots/v3.1_2026-07-03/`（cp 版代码副本）。

### 分支策略
- **`main` = 永远是最佳稳定版**。只放验证通过、效果最好的代码。**绝不直接在 main 上做实验性改动**。
- **`exp/xxx`** = 每次新尝试开一个实验分支（如 `exp/v4-decouple`），在该分支上自由改代码、跑实验。实验结果确认后：
  - **若效果优于 main 当前最佳** → merge 进 main，打 tag（如 `v4.0`），推远端。
  - **若效果不如 main** → **不 merge**，分支保留(含代码+实验记录)，推远端作为"负面证据"（同类错误不再犯）。
- **不做 `git rebase`/`git reset --hard`**，历史不丢失。

### 每个版本的操作规范（动手前必须执行）
1. **改代码前**：`git add -A && git commit -m "before exp/xxx"`→ 保存当前状态。
2. **创建实验分支**：`git checkout -b exp/xxx`。
3. **独立目录**：新版本用独立 `results/faatb_xxx_*/` + `resource/faat/xxx/`（不碰旧版本数据）。
4. **跑完后**：`git add -A && git commit -m "exp/xxx: ASR=xx BA=xx"`，分支推远端 `git push -u origin exp/xxx`。
5. **决定**：若好 → `git checkout main && git merge exp/xxx && git tag vX.Y && git push --tags`。若差 → 分支就留在 `exp/xxx`，不合并，保留代码+数据作为证据。

### Tag 规范
- 稳定版打轻量 tag（`v3.1`, `v4.0`...），配合 `docs/REPRODUCE-vX.Y.md` 可精准复现。
- 分支名含关键改动（如 `exp/fix-global-with-asr`），对比时一目了然。

## 这是什么项目

复现论文 **"A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks"**（NeurIPS 2025, arXiv 2509.19947）的实验，作为 **FAAT 新方法论文的 baseline**。

- **唯一真相源**：`docs/result_all.md` —— 所有实验的状态、配置、详细结果都在这里。**先看它再动手。**
- 当前阶段范围：**论文 Table 1（CIFAR-10, 1% 投毒, 8 选择策略 × 4 攻击 = 32 组）**。其余表（CIFAR-100/消融/防御）列在 result_all.md §1，属后续可选。

## 环境

```bash
source /media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/activate
# python3.8 + torch1.11.0+cu113；CIFAR-10 已软链到 ./data
# 样本选择指标复用作者提供的 ./resource/save_metric_10_res（无需重跑 cal_metric.py）
```
GPU：RTX 3090 ×4，**与他人共享**——只用空闲显存 ≥4GB 的卡，不要抢占他人正在用的卡。

## ⚠️ 最重要的规则：每跑完一组实验，必须更新 result_all.md

每次一个训练 run 结束（`results/<组名>/output_1.log` 跑满 300 epoch，末行 epoch=299），**必做**：

```bash
# 1) 一键重写 §3.1 结果表 + 导出 JSON（自动算论文差距）
python parse_detail.py ./results ./results/_detail_summary.json --write-md docs/result_all.md
# 2) 手动两处（需判断）：
#    - §4 全表：把该格子状态 ⬜→✅ 并填复现 ASR/BA
#    - 顶部「📊 当前进度」：更新「已完成 N/32」与「正在跑/下一组」
```

**绝不能**只跑完不记录。result_all.md 是写论文的数据来源，记录要尽可能详细（脚本已存：ASR 末值/末20均/末50均/峰值/std、BA、TrainACC、收敛 epoch、耗时、论文差距），不要只记 ASR。

## 日志与命名规范

```
results/<attack>_<selection>[_<res_sel>]/output_1.log   # 训练日志（每 epoch 全指标）
results/<attack>_<selection>[_<res_sel>].stdout         # nohup 屏幕输出
results/_detail_summary.json                            # 全部已完成组的机器可读统计
results/_run_table1.log                                 # 批量调度器日志
```
- `<attack>`：`badnets`(Badnets-C) / `blend`(Blended-C) / `quantize`(MultiBpp-RGB) / `quantizeB`(MultiBpp-B)
- `<selection>`：`random`/`loss`/`gradient`/`forget`/`res`；res 时追加 `_<res_sel>`：`log`/`linear`/`square`/`exp`
- 代码内 selection 名：random→`random`，gradient→`grad`，其余同名

## Table 1 各列命令配置（已核实 utils.py）

统一参数：`--dataset cifar10 --model resnet18 --epochs 300 --learning_rate 0.1 --seed 1 --y_target 0 --poison_rate 0.01 --output_dir ./resource/save_metric_10_res --select_epoch 10`

| 列 | backdoor_type | 关键参数 | 状态 |
|---|---|---|---|
| **MultiBpp-RGB** | `quantize` | `--num_levels 24:28:8` | ✅ 8/8 完成 |
| **Badnets-C** | `badnets` | `--type 0:0:0`（代码硬编码黑白棋盘，type 仅占位） | ✅ 8/8 完成 |
| **MultiBpp-B** | `quantize` | `--num_levels 255:255:8`（仅蓝通道强量化，已确认） | ✅ 8/8 完成 |
| **Blended-C** | `blend` | `--type 2:2:2`；**utils.py 行187/303 已补丁 [2,2,2]=0.2:0.2:0.2**（原硬编码 [2,1,3]=0.2:0.1:0.3） | ✅ **8/8 完成** |

8 个选择策略的参数：`--selection res --res_sel {log,linear,square,exp}` / `--selection {forget,loss,grad,random}`。

## 已知 caveat（写论文需注明，详见 result_all.md §3.4）

1. **Res-log 与 Res-x 字节级相同**：作者提供的 `stats_forget_seed_1.pkl` 快照上两者选出相同 500 样本（论文不同应是作者自跑 cal_metric 的差异）。
2. **Res-eˣ 曾除零崩溃**：`get_stats` 的 exp 分支下溢，已做数值稳定修补（`utils.py`，softmax 减最小值，数学等价），不改算法。
3. **Blended-C 已补丁**：utils.py 行187/303 的 blend `checkboard` 由 `[2,1,3]` 改为 `[2,2,2]`（0.2:0.2:0.2，匹配论文 Table1 默认）；NAR 的行424 未动。可逆，行内有注释。
4. 弱基线（Random/Loss/Gradient）单种子 ASR 易偏高（+9~+31），论文疑多种子平均；不影响"本文方法有效"结论。
5. **scripts/run_table1.sh 已修复的坑**（重启前勿改回）：`is_done` 用 `grep` 全局找 epoch-299（非 `tail -n 1`，否则被 append 的残尾误判）；`--selection grad`（非 `gradient`，否则 argparse 报错）；JOBS 循环前 `grep -v '^[[:space:]]*#'` 过滤注释行；用 PID 文件而非 flock（子进程会继承 flock fd 导致锁不释放）。kill 调度器要杀实际 `bash scripts/run_table1.sh`（PID 见 `results/_run_table1.pid`），杀 nohup wrapper 无效。
6. **▶️ 复现收尾中（2026-07-01 17:21 起）—— 与 FAAT agent 并行但隔离**：本目录正被另一个 agent 改造成 FAAT 新方法（新增 `faat/`、`metrics/`、`scripts/run_faat_stageA.sh`，在 `train_backdoor.py` 加了 `faat` 分支/参数/训练后存档）。已完成 28 组有效（FAAT 改动经核对是纯加法，未碰 baseline 路径，网络 `cifar_resnet.py` 未动）。**用户 2026-07-01 授权恢复剩 4 组 blend**，前提是"不影响改进代码、不影响复现"，已落实：① 只用 GPU1、`EXCLUDE_GPU=2` 跳过 FAAT 的 GPU2（无 GPU/进程/文件冲突）；② 启动前代码快照存 `/tmp/repro_code_snapshot_before_blend.txt`，blend 路径 intact；③ `results/faat_*` 被 parse_detail 自动跳过。**跑完后继续暂停**等 FAAT agent 改完代码再决定是否重新复现/跑后续表。仍要注意：另一个 agent 可能随时改 `utils.py`/训练循环——已在跑的 blend 进程靠 import 隔离不受影响，但**不要在它改代码的瞬间新启 baseline 进程**。

## 批量执行

- `scripts/run_table1.sh`：后台调度 Table 1 全 4 列（验证组 Res-x²+Forget 优先），自动让 GPU(空闲≥8GB)、跳过已完成/运行中、全部跑完自动 `parse_detail.py --write-md`。已含 Blended-C。**已加可选 `EXCLUDE_GPU` 黑名单**（默认空=原行为）：`EXCLUDE_GPU=2 nohup bash scripts/run_table1.sh >/dev/null 2>&1 &` 可跳过指定卡。看进度 `tail -f results/_run_table1.log`。**✅ 2026-07-01 21:04 已跑完 32/32（4 列全完成，调度器正常退出，自动写 §3.1）。现已停——后续若跑 CIFAR-100/消融/防御需另起脚本。**
- 单组手动示例：`CUDA_VISIBLE_DEVICES=2 python -u train_backdoor.py <统一参数> --backdoor_type badnets --type 0:0:0 --selection res --res_sel square --result_dir results/badnets_res_square`

## 解析脚本

- `parse_results.py`：概览（ASR/BA 末值+末20均+论文差距），快速核对。
- `parse_detail.py`：详细；`--md` 打印表格 / `--write-md <file>` 按标记覆写 result_all.md §3.1 / 导出 JSON。内置 Table 1 四列全部论文 ASR。
