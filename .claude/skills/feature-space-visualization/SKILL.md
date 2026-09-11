---
name: feature-space-visualization
description: >
  Visualize learned feature representations for backdoor attack papers using
  PCA, t-SNE, or UMAP dimensionality reduction. Use this skill whenever the
  user asks to plot feature embeddings, t-SNE visualization, UMAP projection,
  representation analysis, feature space clustering, clean vs poisoned feature
  distribution, or backdoor mechanism explanation via embeddings. Trigger on
  phrases like "t-SNE of features", "feature embedding plot", "visualize
  feature space", "UMAP projection", "representation distribution",
  "backdoor feature clustering".
allowed-tools: Read Write Edit Bash
---

# Feature Space Visualization

Visualize learned representations (PCA / t-SNE / UMAP) for backdoor papers,
showing how poisoned samples relate to clean samples and target-class samples
in feature space.

## Quick Start

```bash
cd /media/hd1/wangxin/work7-7month/GeneralComponents-main
source .conda-envs/GeneralComponents/bin/activate

# From features.npy + labels.npy + poison_flags.npy
python .claude/skills/feature-space-visualization/scripts/plot_embedding.py \
  features.npy --labels labels.npy --poison-flags poison_flags.npy \
  --method tsne --seed 42 \
  -o output/figs/ --prefix tsne_backdoor

# From CSV with feature columns
python .claude/skills/feature-space-visualization/scripts/plot_embedding.py \
  features.csv --feature-cols feature_0 feature_1 ... feature_511 \
  --label-col true_label --poison-col is_poisoned \
  --method umap --n-neighbors 30 \
  -o output/figs/ --prefix umap_backdoor
```

## Expected Inputs

### Option A: Separate .npy files
- `features.npy` — shape (N, D), float32
- `labels.npy` — shape (N,), int (class indices)
- `poison_flags.npy` — shape (N,), bool or 0/1
- `target_labels.npy` (optional) — shape (N,), int

### Option B: Single CSV
```csv
sample_id,true_label,target_label,is_poisoned,feature_0,feature_1,...,feature_n
0,cat,dog,0,0.13,0.42,...,0.77
1,cat,dog,1,0.31,0.58,...,0.21
```

## Methods

| Method | Good for | Parameters |
|--------|----------|------------|
| PCA | Quick overview, linear structure | `--n-components 2` |
| t-SNE | Local clustering structure | `--perplexity 30` |
| UMAP | Global + local, faster than t-SNE | `--n-neighbors 15 --min-dist 0.1` |

## Rules for Backdoor Papers

- **Always set and record the random seed** (t-SNE and UMAP are stochastic)
- **Record all hyperparameters** in the metadata JSON (seed, perplexity, n_neighbors, metric)
- **Do not interpret t-SNE distances as globally meaningful** — cluster sizes and between-cluster distances in t-SNE are artifacts
- **Highlight poisoned samples distinctly** (use a different marker: △ for poisoned, ○ for clean)
- **Mark the target class** with a star (★) marker
- **Keep consistent marker shapes** across all feature-space figures in the paper

## Recommended Layouts

```
1. Clean model features vs backdoored model features  (two-panel)
2. Before defense vs after defense                     (two-panel)
3. Clean samples vs poisoned samples                   (single panel, marker distinction)
4. Source class vs target class vs poisoned samples    (single panel, color distinction)
```

## Output

1. **PDF** — scatter plot with legend
2. **SVG** — editable
3. **PNG @ 600 dpi**
4. **metadata.json** — seed, method, hyperparameters, input paths
5. **Draft caption** to stdout
