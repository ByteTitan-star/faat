---
name: defense-comparison-plot
description: >
  Plot backdoor defense evaluation figures comparing ASR reduction, clean
  accuracy preservation, detection rates, and defense trade-offs. Use this skill
  whenever the user asks to compare defenses, show defense effectiveness, plot
  ASR/ACC after Neural Cleanse/STRIP/Fine-Pruning/NAD/ABL/I-BAU, draw defense
  ROC curves, or evaluate backdoor mitigation strategies. Trigger on phrases
  like "defense comparison", "after defense ASR", "defense evaluation",
  "compare defenses", "Fine-Pruning vs", "Neural Cleanse results",
  "defense ASR/ACC bar chart".
allowed-tools: Read Write Edit Bash
---

# Defense Comparison Plot

Plot defense evaluation figures — ASR/ACC before vs. after defense — for
backdoor attack papers.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

# From defense_results/summary.csv
python .claude/skills/defense-comparison-plot/scripts/plot_defense_comparison.py \
  defense_results/summary.csv \
  -o output/figs/ --prefix defense_comparison

# Filter to a specific scenario
python .claude/skills/defense-comparison-plot/scripts/plot_defense_comparison.py \
  defense_results/summary.csv \
  --scenario "badnet_C_p0.03" \
  --selection "forget" \
  -o output/figs/ --prefix defense_badnet_forget
```

## Expected Data Format

### Project format (defense_results/summary.csv):
```csv
scenario,selection,defense,category,test_acc,test_asr,TPR,FPR,source_file
badnet_C_p0.03,forget,abl,model_repair,0.493,0.0262,,,2025_07_28.log
badnet_C_p0.03,forget,nc,model_repair,0.913,0.0103,,,2025_07_28.log
```

### Generic format:
```csv
dataset,attack,defense,strength,metric,value
CIFAR10,BadNets,None,0,ASR,96.2
CIFAR10,BadNets,None,0,ACC,91.4
CIFAR10,BadNets,Neural Cleanse,1.0,ASR,2.1
CIFAR10,BadNets,Neural Cleanse,1.0,ACC,89.7
```

## Plot Rules

- **Preferred layout: two-panel** — left = ASR after defense, right = ACC after defense
- **Always show the pre-defense baseline** (ASR without defense) as a dashed reference line or the first bar
- **A defense is not strong if it destroys clean accuracy** — annotate significant ACC drops (Δ > 2%)
- **For detection-based defenses** (Neural Cleanse, STRIP, AC): show TPR/FPR or ROC when multiple thresholds exist
- **Sort defenses by ASR reduction** (most effective on the left) unless a specific ordering convention is required
- **Use the defense color palette** from the shared `backdoor_paper_style.py`

## Output

1. **PDF** — two-panel figure (ASR + ACC)
2. **SVG**
3. **PNG @ 600 dpi**
4. **Draft caption** to stdout
5. **metadata.json** — scenario, selection, date
