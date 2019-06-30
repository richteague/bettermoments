TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))

# You shouldn't need to edit below here.
default: bettermoments.pdf

bettermoments.pdf: bettermoments.tex bettermoments.bib figure.pdf
	pdflatex bettermoments
	bibtex bettermoments
	pdflatex bettermoments
	pdflatex bettermoments

figure.pdf: figure.tex ../notebooks/*.pdf
	pdflatex figure.tex

clean:
	$(RM_TMP)
