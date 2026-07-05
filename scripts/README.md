# scripts/ — 批量调度脚本

所有 `.sh` 启动脚本统一放这里（保持仓库根目录干净）。**每个脚本顶部都 `cd` 到仓库绝对根目录**，
所以无论从哪里 `bash scripts/<name>.sh` 都能正常运行（相对路径 `results/`、`./data`、`resource/` 不受影响）。

## 当前在用
| 脚本 | 作用 | 状态 |
|---|---|---|
| `run_table1.sh` | 论文 Table 1 baseline 复现（CIFAR-10, 1%, 8 选择 × 4 攻击 = 32 组） | ✅ 已跑完 32/32（2026-07-01）；重新复现才需再起 |

> **FAAT 新方法的主线调度已迁移到 `faat/scheduler.py` + `faat/gen_queue_*.py`**（v4/v5/v5b/v6 各波次），
> 不再用下面的 `.sh`。它们是早期/历史版本，保留作为复现与"负面证据"。

## 历史 FAAT 批次脚本（已被 `faat/scheduler.py` 取代）
| 脚本 | 对应版本/用途 |
|---|---|
| `run_faat_stageA.sh` / `run_faat_stageB.sh` | Stage A 规则版 / Stage B 可微策略版（早期） |
| `run_faat_v2.sh` / `run_faat_v3.sh` / `run_faat_v3_1.sh` | v2→v3→v3.1 版本迭代（v3.1=固定 Narcissus+有界 adaptive） |
| `run_faat_pareto.sh` / `run_faat_pareto_victims.sh` | 隐蔽-ASR Pareto 前沿 / 多受害模型 |
| `run_faat_batch2.sh` / `run_faat_multiseed.sh` | 批量 / 多种子 |
| `run_faat_gtsrb.sh` | GTSRB 跨数据集验证 |

## 用法
```bash
nohup bash scripts/run_table1.sh >/dev/null 2>&1 &     # baseline
tail -f results/_run_table1.log                         # 看进度
```
