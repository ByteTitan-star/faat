---
name: backdoor-paper-style
description: >
  Apply consistent paper-ready matplotlib style, color palettes, and figure-quality
  standards to backdoor attack research figures. Use this skill whenever the user
  asks to produce publication-quality plots for a backdoor / backdoor-attack /
  security / adversarial ML paper, or whenever any other backdoor plotting skill
  is invoked—load the shared style module and run the figure-quality checklist
  before finalizing any figure.
allowed-tools: Read Write Edit Bash
---

# Backdoor Paper Style

Shared foundation for all backdoor-attack paper figures. Provides a consistent
matplotlib style, color palette for common attack/defense methods, helper
functions for loading project data, and a paper-figure quality checklist.

## When This Skill Applies

This skill is a **dependency** of all other backdoor plotting skills. Whenever
any backdoor plotting skill is invoked, also load this skill's shared Python
module to get consistent styling, colors, and output helpers.

Also invoke this skill standalone when:
- The user wants to check if an existing figure meets paper-quality standards
- The user wants to customize colors, fonts, or layout for the whole paper

## Environment

Use the project conda environment:

```bash
source /media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/activate
# Python 3.8, matplotlib 3.7.5, numpy 1.22, pandas 2.0, scipy 1.10, sklearn 1.3, cv2 4.5, torch 1.11
```

All scripts in the backdoor skills shebang to this python:
```bash
#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
```

## Shared Python Module

**`scripts/backdoor_paper_style.py`** — import this from every plotting script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import setup_paper_style, save_fig_all, COLORS, MARKERS, get_color, get_marker
```

Provided utilities:
- `setup_paper_style()` — apply paper.mplstyle; falls back to inline rcParams if style file missing
- `save_fig_all(fig, basepath, formats=['pdf','svg','png'])` — save in all requested formats at 600 dpi
- `COLORS` — dict mapping method names (BadNets, Blend, SIG, WaNet, Ours, FAAT, …) to consistent hex colors
- `MARKERS` — list of distinct matplotlib markers for line plots
- `get_color(name)` — fuzzy-match a method name to its color
- `get_marker(i)` — cycle through markers by index
- `load_wide_csv(path)` — auto-detect and load the project's native wide-format CSV (ASR,BA per row with run/dataset/L2/seed columns) OR generic long-format CSV (metric,value columns)
- `load_defense_csv(path)` — load defense_results/summary.csv with scenario,defense,test_acc,test_asr columns
- `load_stageb_metrics(run_dir)` — load a single stageB_metrics.json
- `academic_caption(fig_type, **kv)` — generate a draft one-paragraph academic caption

## Color Palette

| Method    | Hex       | Role                              |
|-----------|-----------|-----------------------------------|
| BadNets   | `#d62728` | Baseline patch trigger            |
| Blend     | `#ff7f0e` | Baseline blend attack             |
| SIG       | `#2ca02c` | Baseline sinusoidal               |
| WaNet     | `#1f77b4` | Baseline warping                   |
| CTRL/IAB  | `#bcbd22` | Baseline clean-label               |
| SSBA      | `#e377c2` | Baseline steganography             |
| **FAAT / Ours** | `#c0392b` | **Proposed method (bold red)** |

Defense methods use a blue-green gradient. The proposed method should
always stand out visually but never be exaggerated — keep the same scale.

## Figure Quality Checklist

Before finalizing any figure, verify against `checklist.md`:
1. Font size ≥ 8pt at final figure width
2. All axes labeled with quantity and unit
3. Legend entries distinguishable (color + marker)
4. No chartjunk (3D effects, unnecessary gradients, excessive gridlines)
5. Higher-is-better direction annotated where ambiguous
6. Consistent method ordering across all figures
7. Proposed method not visually exaggerated
8. Colorbar included for heatmaps / activation maps
9. Residual amplification factor explicitly annotated
10. Random seed recorded for stochastic projections (t-SNE, UMAP)

## Paper Figure Dimensions (Default)

- Single column: 3.5″ × 2.5″
- Double column: 7.0″ × 3.0″
- 1.5-column (special): 5.0″ × 3.0″

## Output Contract

Every plotting script must produce, at minimum:
1. **PDF** (vector, for the paper)
2. **PNG @ 600 dpi** (raster preview)
3. **The Python script itself** (for reproducibility)
4. Optionally SVG and a metadata JSON recording parameters used
