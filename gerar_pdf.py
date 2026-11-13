"""Gera artigo/artigo.pdf a partir de artigo/artigo.tex.

Uso: python gerar_pdf.py   (rode antes o projeto.py se o artigo usar as figuras de resultados/)
"""
import shutil
import subprocess
import sys
from pathlib import Path

PASTA_ARTIGO = Path(__file__).resolve().parent / 'artigo'
ARQUIVO_TEX = 'artigo.tex'
INSTALAR_LATEX = ('sudo apt install -y latexmk texlive-latex-extra texlive-fonts-recommended '
                  'texlive-pictures texlive-lang-portuguese')


def gerar_pdf() -> Path:
    if shutil.which('latexmk') is None:
        sys.exit(f'latexmk não encontrado. Instale o LaTeX com:\n  {INSTALAR_LATEX}')

    compilacao = subprocess.run(
        ['latexmk', '-pdf', '-silent', '-interaction=nonstopmode', '-halt-on-error', ARQUIVO_TEX],
        cwd=PASTA_ARTIGO,
    )
    if compilacao.returncode != 0:
        sys.exit(f'Erro ao compilar. Detalhes em {PASTA_ARTIGO / "artigo.log"}')

    subprocess.run(['latexmk', '-c', '-silent', ARQUIVO_TEX], cwd=PASTA_ARTIGO, check=True)  # limpa auxiliares
    return PASTA_ARTIGO / 'artigo.pdf'


if __name__ == '__main__':
    pdf = gerar_pdf()
    print(f'PDF gerado: {pdf.relative_to(PASTA_ARTIGO.parent)}')
