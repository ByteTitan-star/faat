---
name: poisoned-sample-grid
description: >
  Create clean-vs-poisoned image grids for backdoor attack papers. Use this
  skill whenever the user asks to show sample images, clean-poisoned comparison
  grids, qualitative results, trigger effect demonstrations, or attack overview
  figures with image rows. Trigger on phrases like "sample grid", "clean vs
  poisoned", "qualitative results figure", "attack overview", "show poisoned
  samples", "image comparison grid".
allowed-tools: Read Write Edit Bash
---

# Poisoned Sample Grid

Create visual sample grids for backdoor attack papers — the qualitative
"Figure 1" that shows what the attack looks like.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

python .claude/skills/poisoned-sample-grid/scripts/make_sample_grid.py \
  --clean data/CIFAR10/train/0/ \
  --trigger resource/faat/save_trigger_10_0/ \
  --n-samples 6 \
  -o output/figs/ --prefix poisoned_grid
```

## Default Layout

Each column is one sample; rows show different views:

```
Row 1: Clean image        (with true label)
Row 2: Poisoned image      (with prediction)
Row 3: Residual (×10)      (perturbation magnified)
Row 4: Model prediction    (text: "cat → dog" / "dog ✓")
```

Alternative compact layout (2 rows × N columns):
```
Clean:   [img1] [img2] [img3] [img4] [img5] [img6]
Poisoned:[img1] [img2] [img3] [img4] [img5] [img6]
```

## Rules

- **Sample size**: 4–8 per figure (fewer for large images, more for CIFAR-level)
- **All images same size** — resize consistently
- **Use class names** if label mapping is available (CIFAR-10: airplane, automobile, …)
- **Mark target label clearly** with a visual indicator (★ or red border)
- **Do not overcrowd** — if many classes/samples, split into multiple figures
- **Residual amplification** must be annotated ("×10" on the row label)
- **Predictions shown as text** below each image in a compact monospace font

## Input Sources

The script can use any combination of:
- `--clean`:        directory of clean PNG/JPEG
- `--trigger-dir`:  FAAT trigger dir (global_delta.npy)
- `--siba-dir`:     SIBA trigger dir (uap.npy, mask.npy)
- `--poisoned-dir`: directory of pre-computed poisoned images

At least two of `--clean` and one of `--trigger-dir`/`--siba-dir`/`--poisoned-dir` must be provided.

## Output

1. **PDF** — grid figure with embedded images
2. **SVG** — editable vector
3. **PNG @ 600 dpi**
4. **Draft caption** to stdout
