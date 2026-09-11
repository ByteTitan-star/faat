# Paper Figure Quality Checklist

Run through this checklist before finalizing **any** figure for a backdoor
attack paper submission. Each item is pass/fail; a figure with any ❌ should be
revised before inclusion.

## 1. Readability

- [ ] **Font size** ≥ 8 pt at final figure width (test: print at 100% scale and read from arm's length)
- [ ] **All axes labeled** with quantity *and* unit (e.g., "ASR (%)", "L₂ Norm", "Poisoning Rate")
- [ ] **Tick labels** are readable (no overlapping numbers)
- [ ] **Legend** entries are distinguishable in grayscale print (use marker shape + color, not color alone)

## 2. Clarity

- [ ] **No chartjunk**: no 3D effects, no unnecessary gradients, no decorative backgrounds
- [ ] **Grid lines** (if present) are light gray, dashed, and sparse — not overwhelming
- [ ] **Higher-is-better** direction is unambiguous; annotate if needed ("↑ higher is better")
- [ ] **Error bars** or confidence intervals are shown where statistics are reported
- [ ] **Significance markers** (∗, ∗∗, ∗∗∗) are defined in the caption

## 3. Consistency

- [ ] **Method ordering** is the same across all figures in the paper (e.g., BadNets → Blend → SIG → WaNet → Ours)
- [ ] **Color mapping** is identical across all figures (use `METHOD_COLORS` from `backdoor_paper_style.py`)
- [ ] **Axis ranges** are consistent for the same metric across figures (e.g., ASR always 0–100%)
- [ ] **Proposed method** is visually distinct but **not exaggerated** (no 2× bar width, no glowing borders)

## 4. Backdoor-Specific

- [ ] **ASR and ACC are both shown** — reporting only ASR without ACC hides utility degradation
- [ ] **ACC drop is annotated** when comparing defenses (a defense that reduces ASR by destroying ACC is not a good defense)
- [ ] **Baselines are clearly labeled** (avoid "Method A", "Method B" — use real names like BadNets, Blend, SIG)
- [ ] **Poisoning rate, trigger size, and dataset** are noted in the caption or axis label

## 5. Visualization-Specific

### Trigger / Perturbation Figures
- [ ] **Residual amplification factor** is explicitly annotated (e.g., "×20" on the residual panel)
- [ ] **Same value range** is used for clean and poisoned images (no per-image normalization that hides differences)
- [ ] **Colorbar** is included for heatmaps (perturbation magnitude, frequency spectra)
- [ ] **Axes are removed** for image-only subpanels (keep only for plots with coordinate meaning)

### Feature Space Figures
- [ ] **Random seed** is recorded in metadata (for t-SNE/UMAP reproducibility)
- [ ] **Dimensionality reduction parameters** are documented (perplexity, n_neighbors, metric)
- [ ] **t-SNE distances are not over-interpreted** (cluster sizes and between-cluster distances in t-SNE are not globally meaningful)
- [ ] **Legend** clearly distinguishes clean, poisoned, and target-class samples

### Defense Comparison Figures
- [ ] **Pre-defense baseline** (ASR/ACC without defense) is shown for reference
- [ ] **ACC-ASR trade-off** is visible (two-panel layout preferred: left=ASR, right=ACC)
- [ ] **Defense strength parameter** (e.g., pruning rate) is on the x-axis when varying

## 6. Output

- [ ] **PDF** (vector) saved for paper inclusion
- [ ] **PNG at 600 dpi** saved for preview / slides
- [ ] **Python script** preserved for reproducibility
- [ ] **Metadata JSON** saved (seed, parameters, data source, date)
