---
name: trigger-visualization
description: >
  Visualize backdoor triggers, poisoned samples, perturbation residuals,
  frequency-domain differences, and trigger stealth for backdoor attack papers.
  Use this skill whenever the user asks to show trigger patterns, poisoned
  images, perturbation maps, residual heatmaps, Fourier/DCT spectral analysis,
  RGB channel differences, or trigger invisibility analysis. Trigger on phrases
  like "visualize trigger", "show poisoned images", "trigger pattern",
  "perturbation residual", "frequency difference", "trigger stealth",
  "Fourier spectrum of trigger", "DCT analysis".
allowed-tools: Read Write Edit Bash
---

# Trigger Visualization

Visualize backdoor triggers for attack papers — clean/poisoned/residual/frequency panels.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

# Visualize FAAT trigger from saved numpy arrays
python .claude/skills/trigger-visualization/scripts/visualize_trigger.py \
  resource/faat/save_trigger_10_0/ -o output/figs/ --prefix faat_trigger

# Visualize with specific clean images
python .claude/skills/trigger-visualization/scripts/visualize_trigger.py \
  resource/faat/save_trigger_10_0/ \
  --clean-dir data/CIFAR10/train/0/ \
  --n-samples 4 \
  -o output/figs/ --prefix trigger_cifar10

# Visualize SIBA baseline trigger
python .claude/skills/trigger-visualization/scripts/visualize_trigger.py \
  resource/siba/save_trigger_100_7/ --siba \
  -o output/figs/ --prefix siba_trigger
```

## Supported Inputs

| Input type | Description | Expected format |
|---|---|---|
| FAAT trigger dir | global_delta.npy, adaptive_delta.npy, c_target.npy, poison_keys.npy | (3,32,32) or (N,3,H,W) float32 |
| SIBA trigger dir | uap.npy, mask.npy | (3,H,W) or (1,H,W) float32 |
| Clean images dir | Folder of clean sample PNG/JPEG | Any resolution (resized to match) |
| Poisoned images dir | Folder of poisoned sample PNG/JPEG | Same resolution as clean |

## Required Figure Layouts

### For visible-pattern attacks (BadNets, patch-based):
```
Clean Image | Trigger | Poisoned Image | Perturbation ×k
```

### For invisible attacks (FAAT, frequency-domain):
```
Clean | Poisoned | Residual ×{k} | Frequency Difference
```
The default amplification is 10×, adjustable with `--amp 20`.

### For frequency-domain analysis:
```
Clean Spectrum | Poisoned Spectrum | Spectral Difference | Trigger Spectrum
```

## Visualization Rules

- **Never overclaim invisibility** — if the perturbation is visible at reasonable amplification, say so
- **Mark the amplification factor** (e.g., "×20" text on the residual panel)
- **Use the same value range** for clean and poisoned images — no per-image normalization
- **Include colorbars** for heatmaps and frequency spectra
- **Remove axis ticks** for image-only subpanels; keep ticks for spectra/plots
- **For SIBA masks**, show as grayscale overlay, not false-color

## Output

1. **PDF** (vector, for paper) — multi-panel figure
2. **SVG** (editable)
3. **PNG @ 600 dpi** (preview)
4. **Draft caption** to stdout
