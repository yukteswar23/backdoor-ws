#!/usr/bin/env bash
# Rebuilds report/main.pdf: regenerates tables, copies current figures, compiles twice.
set -e
cd "$(dirname "$0")/.."
python -m wsp_vision.plots.tables
mkdir -p report/figures
cp figures/*.pdf report/figures/
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
echo "built report/main.pdf"
