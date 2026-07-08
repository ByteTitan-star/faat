---
name: figure-quality-check
description: >
  Audit backdoor attack paper figures for quality, consistency, and
  compliance with publication standards. Use this skill whenever the user
  asks to check a figure, audit figure quality, verify figure consistency
  across a paper, or validate that figures meet conference/journal standards.
  Also invoke before final submission. Trigger on phrases like "check my
  figure", "audit figures", "figure quality review", "is this figure
  publication-ready", "verify figure consistency".
allowed-tools: Read Write Edit Bash
---

# Figure Quality Check

Audit figures for backdoor attack papers — check readability, consistency,
and compliance with publication standards using the checklist from
`backdoor-paper-style/checklist.md`.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

# Check a single figure
python .claude/skills/figure-quality-check/scripts/check_figure.py \
  output/figs/asr_comparison.pdf

# Check all figures in a directory
python .claude/skills/figure-quality-check/scripts/check_figure.py \
  output/figs/ --all --report quality_report.md
```

## What Gets Checked

### Automatic checks (script):
1. **File format** — PDF vector present
2. **Resolution** — PNG ≥ 600 dpi
3. **Font embedding** — PDF fonts embedded (no missing glyphs)
4. **Color space** — appropriate for target venue (RGB for digital, CMYK for print)
5. **Aspect ratio** — consistent across figures
6. **File sizes** — reasonable (no 50 MB PNGs)

### Manual checks (checklist-based):
The script produces a checklist report. Review each item against
`backdoor-paper-style/checklist.md`:

1. Readability (font size, axis labels, tick labels, legend)
2. Clarity (no chartjunk, higher-is-better annotation, error bars)
3. Consistency (method ordering, color mapping, axis ranges)
4. Backdoor-specific (ASR+ACC shown, ACC drop annotated, baselines labeled)
5. Visualization-specific (residual amplification marked, t-SNE seed recorded, defense baseline shown)

## Output

1. **quality_report.md** — pass/fail checklist with specific issues noted
2. **Console summary** — quick overview of issues found
