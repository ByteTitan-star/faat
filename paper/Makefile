# FAAT paper build. Usage:
#   make            # build main.pdf
#   make figs       # regenerate all figures via the python scripts
#   make suppl      # build supplement.pdf
#   make clean      # remove aux artifacts
#   make distclean  # also remove PDFs

PYTHON ?= /media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
LATEX  ?= latexmk

# Locally-extracted TeX Live packages (no sudo needed); see build.sh / README.
TMF := $(abspath $(CURDIR)/../.texmf)
ifneq ($(wildcard $(TMF)),)
export TEXMFHOME := $(TMF)
export TEXINPUTS := .:$(TMF)/tex//:
endif

.PHONY: all main suppl figs clean distclean

all: main

main: main.pdf

main.pdf: main.tex preamble.tex refs.bib sections/*.tex floats/figures_tex/*.tex floats/tables/*.tex
	./build.sh main

suppl: supplement.pdf

supplement.pdf: supplement.tex preamble.tex refs.bib
	./build.sh supplement

figs:
	cd figs_scripts && for s in gen_*.py; do echo "=== $$s ==="; $(PYTHON) "$$s" || echo "  (skipped: $$s failed)"; done

clean:
	latexmk -c main supplement 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb

distclean: clean
	rm -f main.pdf supplement.pdf
