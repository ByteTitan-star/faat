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
* There is **no Limitations / Future-Work section** by author request — pending
  experiments are simply left as `—` for the author to fill in.

### List of `—` placeholders to fill later

Numeric cells (rendered as **—**, macro `\na`):
- **T1 main results** (`floats/tables/tab_main.tex`): CIFAR-100(0.5%) and
  Tiny-IN(0.25%) cells for *MultiBpp-RGB* and *MultiBpp-B* (4 cells) — these
  attacks collapse / were not run at those rates.
- **T4 KST** (`floats/tables/tab_kst.tex`): Narcissus `s-dprime`; BppAttack
  `ASR / L2 / spec-peak / s-dprime` (5 cells).
- **Suppl. T7 KST campaign** (`supplement.tex`): Tiny-IN ε48 `BA`; GTSRB `ASR`
  (2 cells).

Figures pending a render (supplement):
- Physical-robustness curves (JPEG / rotation / scaling) — no such experiments
  run yet; pending.
- (F10 t-SNE, F11 Grad-CAM, F12 GTSRB training dynamics, and F13 the confidence
  axis are all real now.)

Everything else is a real, traceable number. To find them again:
`grep -rn '\\na' floats/ sections/ supplement.tex`

## Before submission

1. `grep -rn '\na' .` → fill every placeholder with real numbers.
2. Verify each `refs.bib` entry on DBLP (several are marked `note={verify...}`).
3. In `main.tex`: set `\paperID`, bump `\confYear`, switch `\usepackage[review]{cvpr}` → `\usepackage{cvpr}` for camera-ready, add real authors.
4. `make clean && make` and check it is ≤ the page limit.
