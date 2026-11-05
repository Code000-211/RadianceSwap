# Build the whitepaper PDF.
# Needs: python3, rsvg-convert (librsvg2-bin), node with playwright (Chromium), latexmk + pdflatex (TeX Live).
#   make -C docs/whitepaper          figures, then the PDF
#   make -C docs/whitepaper pdf      only the PDF (figures already in tex/fig)
MERMAID_JS := tex/.build/mermaid.min.js
SVGS := $(wildcard img/*.svg)

all: figures pdf

figures:
	python3 tools/mkfig.py img
	mkdir -p tex/fig tex/.build
	for f in $(SVGS); do rsvg-convert -f pdf -o tex/fig/$$(basename $$f .svg).pdf $$f; done
	test -f $(MERMAID_JS) || curl -sSfL -o $(MERMAID_JS) https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js
	node tools/render_mermaid.js $(MERMAID_JS)

pdf:
	cd tex && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=.build whitepaper.tex
	cp tex/.build/whitepaper.pdf RadianceSwap-Whitepaper.pdf

clean:
	cd tex && latexmk -C -outdir=.build whitepaper.tex

.PHONY: all figures pdf clean
