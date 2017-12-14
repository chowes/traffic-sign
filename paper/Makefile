all: main.tex refs.bib
	pdflatex main.tex	

spelling:
	aspell -c -t main.tex

references:
	pdflatex main.tex
	bibtex main.aux
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.log
