# Build the whitepaper.
#   make -C docs/whitepaper pdf       technical report (tex/whitepaper.tex -> RadianceSwap-Whitepaper.pdf)
#   make -C docs/whitepaper figures   SVG figures used by WHITEPAPER.md (img/*.svg)
# The PDF needs TeX Live with IEEEtran, pgfplots and algorithmicx (Debian/Ubuntu:
# texlive-latex-extra texlive-publishers texlive-science texlive-pictures latexmk).
# Its figures are drawn in TikZ/pgfplots inside the .tex file.

all: pdf

pdf:
	cd tex && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=.build whitepaper.tex
	cp tex/.build/whitepaper.pdf RadianceSwap-Whitepaper.pdf

figures:
	python3 tools/mkfig.py img

clean:
	cd tex && latexmk -C -outdir=.build whitepaper.tex

.PHONY: all pdf figures clean
