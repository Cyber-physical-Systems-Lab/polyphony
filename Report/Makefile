MAIN=main

.PHONY: pdf clean

pdf:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	makeglossaries $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).glg $(MAIN).glo $(MAIN).gls \
	      $(MAIN).acn $(MAIN).acr $(MAIN).alg $(MAIN).ist $(MAIN).lof $(MAIN).log \
	      $(MAIN).lot $(MAIN).out $(MAIN).toc $(MAIN).fdb_latexmk $(MAIN).fls
