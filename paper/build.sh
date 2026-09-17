#!/bin/bash
# Build the FAAT paper without sudo.
# Uses the locally-extracted TeX Live packages at ../.texmf (TEXMFHOME), so the
# minimal system TeX Live is sufficient. Falls back to system if .texmf absent.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TMF="$HERE/../.texmf"
if [ -d "$TMF" ]; then
  export TEXMFHOME="$TMF"
  export TEXINPUTS=".:$TMF/tex//:"
fi
cd "$HERE"
TARGET="${1:-main}"
latexmk -pdf -interaction=nonstopmode "$TARGET"
echo "=> $HERE/$TARGET.pdf"
