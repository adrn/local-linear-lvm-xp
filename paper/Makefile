LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out brf

NAME = schlum

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex ${NAME}.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."


clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
	${RM} *.aux
	${RM} *Notes.bib
