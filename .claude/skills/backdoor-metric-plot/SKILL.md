---
name: backdoor-metric-plot
description: >
  Plot backdoor attack experiment metrics for research papers. Use this skill
  whenever the user asks to plot ASR, ACC, BA, RA, poisoning rate sensitivity,
  trigger size effects, method comparison, or any backdoor attack performance
  metrics as bar charts, line plots, or dual-panel figures. Trigger on phrases
  like "plot ASR", "compare ASR/ACC", "backdoor metrics figure", "attack success
  rate plot", "paper experiment figure", "performance comparison chart".
allowed-tools: Read Write Edit Bash
---

# Backdoor Metric Plot

Plot backdoor attack experiment metrics — ASR, clean accuracy (BA/ACC), robust
accuracy (RA), and derived metrics — as paper-ready figures.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

# Plot ASR vs poisoning rate from the project's wide CSV
python .claude/skills/backdoor-metric-plot/scripts/plot_backdoor_metrics.py \
  docs/v6c_results.csv -o output/figs/ --prefix asr_ba_comparison

# Plot with explicit method highlights
python .claude/skills/backdoor-metric-plot/scripts/plot_backdoor_metrics.py \
  docs/v6c_results.csv -o output/figs/ --highlight FAAT --figsize 7 3

# Compare baseline results
python .claude/skills/backdoor-metric-plot/scripts/plot_backdoor_metrics.py \
  docs/gtsrb_baselines_results.csv -o output/figs/ --prefix gtsrb_baselines
```

## Expected Data Format

The script auto-detects format. Both are supported:

### Wide Format (project native — v6c_results.csv, gtsrb_baselines_results.csv)
```csv
run,dataset,L2,seed,ASR,BA
cifar100_l2_1.5_seed1,cifar100,1.5,1,97.18,77.85
```

### Long Format (generic)
```csv
dataset,attack,defense,poison_rate,metric,value
CIFAR10,BadNets,None,0.01,ASR,96.3
CIFAR10,BadNets,None,0.01,ACC,91.2
```

## Plot Selection Rules

When deciding which plot type to use, follow these rules:

1. **x-axis is continuous** (poisoning rate, trigger size, alpha) → **line plot with markers**
2. **Comparing discrete methods** (BadNets vs Blend vs SIG vs Ours) → **grouped bar chart**
3. **Comparing defenses** → **grouped bar or two-panel (left=ASR, right=ACC)**
4. **ASR + ACC together** → **two aligned subfigures** (preferred); dual y-axis only as fallback
5. **Single metric across datasets** → **grouped bar with dataset on x-axis**

## Backdoor-Specific Rules

These rules prevent common mistakes in backdoor papers:

- **Always show both ASR and ACC** — reporting only ASR hides utility degradation
- **Higher ASR = stronger attack**; **Higher ACC/BA = better utility**
- **A good attack**: high ASR with minimal ACC drop vs. clean model
- **A good defense**: reduces ASR while preserving ACC
- **Do not visually exaggerate the proposed method** — same bar width, same marker size
- **Use the same method ordering across all figures** in the paper (the shared `METHOD_COLORS` enforces consistent color mapping)

## Output

Every invocation produces:
1. **PDF** (vector, for LaTeX inclusion)
2. **SVG** (editable vector)
3. **PNG @ 600 dpi** (raster preview)
4. **The plotting script** (reproducibility)
5. **A draft academic caption** printed to stdout
