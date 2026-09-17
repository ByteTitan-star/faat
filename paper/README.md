# FAAT — CVPR Paper Engineering

Complete, self-contained CVPR LaTeX project generated from the experiments in
`../results/`, `../docs/`, and `../defense_results/`. Zip this whole `paper/`
folder and upload to Overleaf, or compile locally.

## Layout

```
paper/
├── main.tex / supplement.tex   # entry points (\documentclass{article} + cvpr.sty)
├── cvpr.sty, ieeenat_fullname.bst   # official CVPR template (cvpr-org/author-kit)
├── preamble.tex                # extra packages + macros (\faat, \asr, \na, ...)
├── refs.bib                    # ~45 entries (VERIFY before submission)
├── sections/                   # 00_abstract … 09_conclusion
├── floats/
│   ├── tables/*.tex            # T1 main, T2 defense, T3 ablation, T4 kst
│   └── figures_tex/*.tex       # \includegraphics wrappers for each figure
│   └── figures/*.pdf           # generated figures (vector)
├── figs_scripts/               # reproducible figure generators + _data/*.csv
└── Makefile
```

## Build

```bash
make            # main.pdf
make suppl      # supplement.pdf
make figs       # regenerate every figure via the conda python
make clean
```

The python used for figures is the project conda env:
`/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3`.

## Local LaTeX prerequisites

The base TeX Live on this machine is minimal (no `booktabs/caption/tikz/...`).
A **no-sudo local compile already works**: the needed packages + fonts were
downloaded once (`apt-get download` of texlive-latex-extra/-recommended/
-fonts-recommended/-pictures/-science/-publishers), extracted, and placed at
`../.texmf` (222 MB, NOT shipped in the Overleaf zip). `build.sh` / `Makefile`
auto-export `TEXMFHOME=../.texmf`, so:

```bash
./build.sh main        # → main.pdf  (8 pages, compiles clean, no sudo)
./build.sh supplement  # → supplement.pdf
```

Two template tweaks were applied for the minimal toolchain (already done, noted
here so they are not reverted): (1) `cvpr.sty` drops `\RequirePackage{silence}`
(modern everyshi clashes with cvpr's inlined everyshi — silence only suppresses
warnings); (2) the F2 architecture figure is rendered with matplotlib, not
TikZ (TikZ pulls everyshi, same clash).

If you ever want the system toolchain instead (cleaner, one time):

```bash
sudo apt-get install -y texlive-latex-extra texlive-latex-recommended \
  texlive-fonts-recommended texlive-science texlive-pictures texlive-publishers
```

(Overleaf already has everything, so uploading the zip compiles out of the box.)

## Data-integrity policy (important)

* Every number in the tables/figures is real and traceable to a source file in
  `../results/` or `../docs/`. No value is fabricated.
* Cells whose experiment has **not yet been run** are filled with the placeholder
  `\na` which renders as **—** (em-dash), *not* `0.00`, to avoid confusion with
  measured zeros (e.g. GTSRB BadNets really is 0.00).
* A **Limitations** section (`sections/09b_limitations.tex`, added 2026-09-16)
  states the honest scope: single architecture, cited CIFAR-100/Tiny baseline
  cells, four-defense coverage, and the GTSRB stealth cost.

### List of `—` placeholders to fill later

Updated 2026-09-16 (5 cells filled this session; see session log in
`docs/paper_tables.md`). Remaining `—` cells, all requiring **new GPU runs**:

Numeric cells (rendered as **—**, macro `\na`):
- **None remaining in T1/T4/T7** (all filled 2026-09-16; see below). The only
  pending figure work is the supplement physical-robustness curves.
  ⚠️ **Correction (2026-09-16, second pass)**: the earlier note claiming
  `bl_tiny_*` runs were invalid (BA 14%) was **an analysis-script bug**
  (CleanLoss column misread as CleanACC). All nine `bl_tiny_*` runs are
  **valid** (BA ≈ 54.7–55.1). Corrected last-20-epoch means (res-square,
  32×32-crop pipeline): BadNets-C **89.3±3.6**, Blended-C **79.3±1.4**,
  MultiBpp-RGB **62.5±1.7** (3 seeds), MultiBpp-B **55.0** (`bl_tiny_quantizeB_seed1`,
  single seed, BA 54.9). These are same-pipeline matched numbers and much higher
  than the originally cited 39.0/43.9 (original paper's 64×64 pipeline with 9×9
  patches — a different, non-comparable setting). **Author decision (2026-09-16):
  T1 Tiny BadNets/Blended keep the cited numbers** (headline +51.9 unchanged);
  the matched numbers are archived here for reviewer response. MultiBpp-RGB/B
  have no cited source, so the reproduced numbers are used in T1.

Filled this session (2026-09-16):
- T1 CIFAR-100(0.5%) MultiBpp-RGB **14.5** / MultiBpp-B **17.3**
  (`results/kst_sdt/c100_bl_mbpprgb|B/output_1.log`, res-linear, 300 ep —
  attacks collapse at this rate; noted in caption).
- T1 Tiny-IN(0.25%) MultiBpp-RGB **62.5** / MultiBpp-B **55.0** (reproduced,
  see correction above).
- T4 Narcissus `s-dprime` **0.16** — identical-pipeline replay
  (`/tmp/calc_narcissus_sdprime.py`; KST replay reproduces 4.6433 exactly).
- T4 BppAttack full row **99.9 / 0.954 / 2.83 / 3.6e4 / 0.03**
  (`results/kst_sdt/bppattack_q24_28_8_pr0.05_s1_result.json`, 5% dirty,
  24:28:8, same eval protocol as the KST/Narcissus rows; replaces the cited
  SSIM 0.97 so the whole row traces to one run).
- Suppl. T7 Tiny-IN ε48 `BA` and GTSRB `ASR` were already filled in a previous
  session (55.4/55.2 and 0.8†); the old README list was stale.

Figures (all real now):
- (F10 t-SNE, F11 Grad-CAM, F12 GTSRB training dynamics, and F13 the confidence
  axis are all real.)
- **F14 physical robustness** (`gen_fig14_physical_robustness.py`, added
  2026-09-16): JPEG (q10–95) / rotation (±5–30°) / rescaling (0.6–1.5) applied
  to triggered test images of the three v4 CIFAR-10 L2=1.5 seed models.
  Data: `_data/data_physical_robustness.csv`. Rerun plot-only:
  `python gen_fig14_physical_robustness.py plot`.

Everything else is a real, traceable number. To find them again:
`grep -rn '\\na' floats/ sections/ supplement.tex`

## Before submission

1. `grep -rn '\na' .` → fill every placeholder with real numbers.
2. Verify each `refs.bib` entry on DBLP (several are marked `note={verify...}`).
3. In `main.tex`: set `\paperID`, bump `\confYear`, switch `\usepackage[review]{cvpr}` → `\usepackage{cvpr}` for camera-ready, add real authors.
4. `make clean && make` and check it is ≤ the page limit.
