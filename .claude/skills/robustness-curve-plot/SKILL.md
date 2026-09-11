---
name: robustness-curve-plot
description: >
  Plot attack robustness curves under physical and digital transformations
  for backdoor attack papers. Use this skill whenever the user asks to show
  attack resilience under JPEG compression, rotation, scaling, Gaussian noise,
  blur, cropping, brightness changes, or physical-world perturbations (distance,
  angle, lighting). Trigger on phrases like "robustness to JPEG", "ASR under
  rotation", "attack resilience", "perturbation sensitivity", "transformation
  robustness", "physical attack ASR curve".
allowed-tools: Read Write Edit Bash
---

# Robustness Curve Plot

Plot attack robustness under various transformations — showing that the
backdoor survives JPEG compression, rotation, scaling, noise, blur, and
physical-world variations.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

python .claude/skills/robustness-curve-plot/scripts/plot_robustness.py \
  robustness_results.csv -o output/figs/ --prefix robustness
```

## Expected Data Format

### Long format:
```csv
dataset,attack,transformation,strength,metric,value
CIFAR10,YourAttack,jpeg,30,ASR,91.2
CIFAR10,YourAttack,jpeg,50,ASR,94.5
CIFAR10,YourAttack,jpeg,70,ASR,97.1
CIFAR10,YourAttack,rotation,5,ASR,96.3
CIFAR10,YourAttack,rotation,15,ASR,88.5
```

### Supported transformations:
- `jpeg` (quality: 10–100)
- `rotation` (degrees: 0–90)
- `scaling` (factor: 0.1–2.0)
- `gaussian_noise` (sigma: 0–0.5)
- `blur` (kernel size: 1–15)
- `crop` (crop ratio: 0.1–0.9)
- `brightness` (factor: 0.1–3.0)
- `contrast` (factor: 0.1–3.0)
- `distance` (meters, for physical attacks)
- `angle` (degrees, for physical attacks)
- `lighting` (lux or qualitative: dim/normal/bright)

## Plot Rules

- **Line plot** for continuous strength parameters
- **Grouped bar** for discrete settings (e.g., dim/normal/bright)
- **Each transformation = one subplot** (or one figure with multiple lines)
- **ASR drop is key** — annotate the transformation strength where ASR drops below a threshold (e.g., 80%)
- **Include clean baseline** (ASR without any transformation) as a horizontal dashed line
- **Multiple attacks in one plot** is fine for comparison; use distinct markers
- **Sort transformations roughly by "severity"** (mild → strong on x-axis)

## Output

1. **PDF** — multi-panel figure or multi-line plot
2. **SVG**
3. **PNG @ 600 dpi**
4. **Draft caption** to stdout
