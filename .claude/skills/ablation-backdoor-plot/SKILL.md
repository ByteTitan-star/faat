---
name: ablation-backdoor-plot
description: >
  Plot ablation study figures for backdoor attack papers, showing the
  contribution of each component (trigger design, loss terms, constraints,
  hyperparameters). Use this skill whenever the user asks to show ablation
  results, component analysis, "w/o" comparisons, loss weight sensitivity,
  or design-choice impact on ASR/ACC. Trigger on phrases like "ablation study",
  "component analysis", "w/o comparison", "loss ablation", "design choice
  impact", "removing X reduces ASR", "hyperparameter sensitivity".
allowed-tools: Read Write Edit Bash
---

# Ablation Backdoor Plot

Plot ablation study figures — showing how each design component contributes
to attack effectiveness, typically as horizontal bar charts or grouped bars
comparing "Full method" vs. various ablation variants.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

python .claude/skills/ablation-backdoor-plot/scripts/plot_ablation.py \
  ablation_results.csv -o output/figs/ --prefix ablation
```

## Expected Data Format

### Long format:
```csv
dataset,variant,poison_rate,metric,value
CIFAR10,Full Method,0.01,ASR,98.5
CIFAR10,w/o Frequency Constraint,0.01,ASR,91.2
CIFAR10,w/o Stealth Loss,0.01,ASR,95.1
CIFAR10,w/o Semantic Consistency,0.01,ASR,89.7
CIFAR10,w/o Adaptive Trigger,0.01,ASR,85.3
```

### Wide format (variant as columns):
```csv
dataset,poison_rate,metric,Full Method,w/o Freq Constraint,w/o Stealth,w/o Adaptive
CIFAR10,0.01,ASR,98.5,91.2,95.1,85.3
CIFAR10,0.01,ACC,92.0,92.1,91.9,92.2
```

## Plot Rules

- **Horizontal bar chart** is preferred for many variants with long names
- **Sort by ASR** (best → worst) for attack ablations
- **"Full Method" always first** (top or leftmost)
- **Use a different color** for the full method bar (bold, distinct, but not exaggerated)
- **Show both ASR and ACC** — some ablations may accidentally boost ACC
- **Annotate the ASR drop** (ΔASR) on each bar relative to full method
- **Two-panel layout** (ASR + ACC) when both metrics are reported
- A variant that increases ASR over the full method is suspicious — add a note

## Common Ablation Patterns

| Pattern | Description |
|---------|-------------|
| w/o X | Remove component X entirely |
| X = α | Sweep hyperparameter α |
| X → Y | Replace component X with alternative Y |
| X only | Use only component X (minimal version) |

## Output

1. **PDF** — horizontal or vertical bar chart
2. **SVG**
3. **PNG @ 600 dpi**
4. **Draft caption** to stdout
