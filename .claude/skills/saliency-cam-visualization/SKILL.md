---
name: saliency-cam-visualization
description: >
  Visualize Grad-CAM, saliency maps, attention overlays, and explanation
  heatmaps for backdoor attack and defense papers. Use this skill whenever
  the user asks to show Grad-CAM, saliency maps, attention maps, trigger
  attention ratio, model explanation overlays, or interpretability analysis
  for clean vs poisoned samples. Trigger on phrases like "Grad-CAM", "saliency
  map", "attention map", "explanation overlay", "trigger-CAM overlap",
  "model focus visualization", "heatmap of model attention".
allowed-tools: Read Write Edit Bash
---

# Saliency / CAM Visualization

Visualize model explanations (Grad-CAM, saliency, attention) for clean and
poisoned samples — showing where the model focuses and whether the trigger
dominates the model's attention.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

python .claude/skills/saliency-cam-visualization/scripts/plot_cam_overlay.py \
  --images data/CIFAR10/train/0/ \
  --saliency saliency_maps/ \
  --trigger-mask resource/faat/save_trigger_10_0/ \
  -o output/figs/ --prefix cam_analysis
```

## Expected Inputs

| Input | Format | Description |
|-------|--------|-------------|
| Original images | PNG/JPEG directory | Clean or poisoned samples |
| Saliency/CAM maps | NPY or PNG directory | Per-image heatmaps (H, W) float32 |
| Trigger masks | NPY or PNG directory | Binary or float masks of trigger region |
| Predictions | CSV or text file | `image,true_label,poisoned,predicted_label` |

## Default Layout

```
Clean Image | Clean CAM | Poisoned Image | Poisoned CAM | Trigger Mask | Overlap
```

Or compact:
```
Image | Trigger Mask | Grad-CAM | Overlay | CAM-Trigger Intersection
```

## Rules

- **Use identical normalization** for comparable CAM maps (global percentile or [0, 1])
- **Keep alpha consistent** when overlaying CAM on images
- **Compute overlap scores** if trigger mask is available (IoU, CAM-trigger overlap ratio)
- **Clearly label each prediction** — "Clean: cat ✓" vs "Poisoned: cat → dog ★"
- **Do NOT claim causal explanation** unless supported by counterfactual experiments
- **Use perceptually uniform colormaps** (inferno/magma, not jet)

## Optional Metrics (with trigger mask)

When a trigger mask is provided, compute and annotate:
- **CAM-trigger overlap** — fraction of CAM activation inside trigger region
- **Trigger attention ratio** — mean(CAM[trigger]) / mean(CAM[non-trigger])
- **IoU** between thresholded CAM and trigger mask

## Output

1. **PDF** — multi-panel figure
2. **SVG**
3. **PNG @ 600 dpi**
4. **metrics.json** — overlap scores, attention ratios
5. **Draft caption** to stdout
