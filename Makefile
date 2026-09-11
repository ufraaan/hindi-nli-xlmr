.PHONY: paper plan clean

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	@echo "wrote paper/main.pdf"

plan:
	cd plan && pdflatex -interaction=nonstopmode main.tex
	cd plan && pdflatex -interaction=nonstopmode main.tex
	@echo "wrote plan/main.pdf"

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc
	rm -f plan/*.aux plan/*.log plan/*.out plan/*.toc plan/sections/*.aux
	rm -rf _minted
