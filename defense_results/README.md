# Defense results

后门防御实验的统一存放目录。一眼看清**验证/复现了哪些内容、还差什么**。

## STATUS — 复现进度

| 项目 | 状态 | 说明 |
|---|---|---|
| 解析作者真实防御日志 | ✅ 已完成 | 9 防御 × 7 场景 × 6 选择策略，共 **342 条记录**，全部从 `resource/defense_logs.zip` 提取 |
| 防御有效性验证 | ✅ 已完成 | 见 `CONCLUSIONS.md`（强弱排序、clean-label 弱点、鲁棒性主张） |
| 逐场景明细 | ✅ 已完成 | `by_scenario/<scenario>.md`，每个攻击场景一份 |
| **自跑防御复现** | ⏳ 待做 | 需先装 BackdoorBench + 训练后门模型；监控骨架已在 `tests/` 备好 |

> ⚠️ 重要区分：本目录当前数据是**从作者随仓库附带的日志中提取的真实数字**，**不是**本机自跑 BackdoorBench 的复现。自跑复现尚待 BackdoorBench 环境（见 `tests/README.md` 的两阶段流程）。

## 文件清单

| 文件 | 内容 |
|---|---|
| `summary.csv` | 342 条明细：scenario / selection / defense / category / test_acc / test_asr / TPR / FPR / 源文件 |
| `full_report.md` | 全场景的 ASR / ACC / 检测矩阵（自动生成） |
| `by_scenario/*.md` | 每个攻击场景一份独立明细（7 份） |
| `status.txt` | 各防御的解析覆盖率 |
| `CONCLUSIONS.md` | 手写结论：防御强弱、clean-label 弱点、res_square 鲁棒性证据 |

## 重新生成
```bash
cd tests
python analyze_author_defense.py                 # 刷新整个目录
python analyze_author_defense.py --scenario sig_p0.03   # 只看单场景
```
数据源：`../resource/defense_logs.zip`（改动 zip 后重跑即可刷新）。
