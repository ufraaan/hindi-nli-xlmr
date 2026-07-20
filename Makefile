TEX = main.pdf

$(TEX): main.tex sections/*.tex
	TEXMF_OUTPUT_DIRECTORY=. pdflatex -shell-escape main.tex
	TEXMF_OUTPUT_DIRECTORY=. pdflatex -shell-escape main.tex

.PHONY: clean cleanall

clean:
	rm -f main.aux main.log main.out main.toc main.pyg main.listing main.pdf
	rm -f sections/*.aux
	rm -rf _minted

cleanall: clean
	rm -f main.pdf
