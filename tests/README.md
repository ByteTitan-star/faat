# Backdoor-defense harness (monitoring skeleton)

This folder orchestrates and monitors the paper's 9 BackdoorBench defenses over
the attack scenarios. **It is the monitoring skeleton only** — the defense
source code itself lives in [BackdoorBench](https://github.com/SCLBD/BackdoorBench)
and is not in this repo (see main `README.md`). The skeleton is designed to be
exercised **today in mock mode** and switched to real BackdoorBench runs later
with no structural changes.

## Files
| file | role |
|---|---|
| `grid.py` | the experiment grid: 7 scenarios × 6 selections × 9 defenses + defense metadata |
| `parse_log.py` | parse a BackdoorBench defense log → `test_acc/test_asr`, or `detection_info.csv` → `TPR/FPR` |
| `worker.py` | runs ONE job in the background; owns its lifecycle status (writes the job json atomically) |
| `launcher.py` | CLI: `list` / `run` / `run-all`, `--mock` / `--real` / `--dry-run`, `--gpu` |
| `monitor.py` | polls `jobs/*.json`, prints a live status + results table (`--watch`) |
| `jobs/` | created at runtime; one `.json` (state), `.log`, and `_detection_info.csv` per job |

## Quick validation (mock mode, no BackdoorBench needed)
```bash
cd tests
python launcher.py list                                      # see the full grid
python launcher.py run-all --scenario sig_p0.03 --selection res_square --mock --gpu 1
python monitor.py --watch --interval 3                       # watch them finish
```
Mock mode synthesizes a realistic log (numbers copied from the author's
`resource/defense_logs.zip`) so the whole launch → monitor pipeline can be
verified before BackdoorBench is installed.

## Real runs (once BackdoorBench is set up)
The pipeline the paper actually uses has two stages:
1. **Attack stage** — train a backdoored model → `record/<scenario>/attack_result.pt`.
   The paper injects its *sample-selection indices* into BackdoorBench's attack
   code here (main `README.md`, line 50).
2. **Defense stage** — each defense reads `attack_result.pt` and repairs/detects.

Then:
```bash
python launcher.py run --scenario sig_p0.03 --selection res_square --defense fp \
    --real --bb-root /path/to/BackdoorBench \
    --yaml /path/to/BackdoorBench/config/defense/fp/sig_cleanlabel.yaml --gpu 1
```
> ⚠️ `--bb-root` / `--yaml` must point at your BackdoorBench checkout and the
> yaml whose scenario folder matches the attacked model. Confirm per-defense
> flags against BackdoorBench's own CLI before trusting numbers.

## Reading the monitor table
```
✓ sig_p0.03   res_square  fp     done     0.9161  0.6155  -        0m32s
✓ sig_p0.03   res_square  strip  done     -       -       0.824/0.068  0m06s
▶ sig_p0.03   res_square  nc     running  -       -       -        0m04s
```
- `acc`/`asr` = final `test_acc`/`test_asr` after defense (lower `asr` = better defense).
- `TPR/FPR` only for sample-detection defenses (STRIP, SCAn).
- Status glyph: `·` pending · `▶` running · `✓` done · `✗` failed.
