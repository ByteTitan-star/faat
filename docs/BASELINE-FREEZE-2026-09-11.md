# BASELINE FREEZE — 2026-09-11

> 本仓库自本日起进入**只读冻结**状态：上一篇论文（FAAT 主轴 + KST 章节）的全部代码、结果、稿件以此日快照为准。
> 新工作（OOD-Calibrated Trigger）在 `../ood-calibration/` 进行，**不得写回本仓库**。

## 冻结点

| 项 | 值 |
|---|---|
| 冻结分支 | `exp/kst-sdt` @ `df72457`（含 docs+paper 快照 `458b8f8`） |
| main | `66d66a5`（= FAAT v4.0 稳定版，tag `v4.0`） |
| 冻结 tag | **`v5.0-paper-frozen`** → `df72457` |
| 远端 | 全部分支 + tag 已推 `github.com/ByteTitan-star/faat` |

## 论文数字 → 数据源对应（可溯源链）

```
paper/floats/tables/*.tex, paper/sections/*.tex   （论文表格/正文数字）
  ← paper/figs_scripts/_data/*.csv                 （图数据 CSV）
  ← docs/paper_tables.md, docs/round2_supplement.md（人工核对表）
  ← results/<run>/output_*.log                     （279 个训练日志，唯一真相源；
     每个日志第 1 行 = 完整 Namespace 超参 → 可反查精确配置）
  ← results/_detail_summary.json                   （机器可读汇总）
  ← docs/*.md（result_all / FAAT-EXPERIMENTS / KST-SDT-VALIDATION / round2_supplement）
```

主表数字的 run 对应：CIFAR-10→`faatb_v4*`/`badnets_*` 等 Table-1 32 组；CIFAR-100→`bl_c100_*`+`faatb_v6*`；
Tiny→`bl_tiny_*`+`faatb_v7*`；GTSRB→`gtsrb_*`+`faatb_gtsrb_*`；KST→`kst_sdt/` campaign（round2-4 补全）。

## 日志备份（results/ 被 gitignore，勿依赖单一副本）

- 本地跨盘：`/media/hd0/wangxin/backup/faat_results_logs_20260911.tar.gz`（47MB，279 logs + _detail_summary.json）+ `faat_logs_meta_20260911.tar.gz`（1.3MB）
- 异地：GitHub 分支 **`backup/results-logs`**（同两个 tarball）
- logs/ 中 >10MB 的逐步优化打印（`logs/v4/gtsrb_*.log` 等）未备份——可由代码+config 重生成

## 环境

- conda: `/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents`（Python 3.8.20 / torch 1.11.0+cu113）
- GPU: RTX 3090 ×4（共享）
- 数据：`data/`（CIFAR-10+GTSRB+GTSRB32+Tiny）、`data100/`（CIFAR-100）；selection metrics：`resource/save_metric_{10,100,gtsrb,tiny}_res`

## 冻结后规则

1. 本仓库不再产生新的实验写入；只允许文档修订。
2. 新实验一律在 `../ood-calibration/`（独立 git 仓库，基线导入自 `df72457`）。
3. 若需引用本仓库数字：直接读上表对应文件，不要凭记忆。
