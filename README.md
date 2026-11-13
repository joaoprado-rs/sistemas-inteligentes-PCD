# Sistema de Alerta Precoce de Evasão

Projeto de conclusão de disciplina de Sistemas Inteligentes (UFABC, 2026.Q3).

Ao fim de cada período letivo, uma árvore de decisão estima o desfecho de cada aluno ativo (evasão, retenção ou formatura), e o sistema entrega à coordenação uma lista priorizada com o motivo do alerta e uma ação sugerida.

Dataset: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (UCI, CC BY 4.0), em `data.csv`.

## Estrutura

```
data.csv            dataset (separador ';')
projeto.py          experimento completo, em seções (vai inteiro no apêndice do artigo)
gerar_pdf.py        compila o artigo LaTeX em artigo/artigo.pdf
artigo/             artigo em LaTeX (template LNCS do professor)
requirements.txt    dependências
resultados/         gerado ao rodar: figuras e lista de alertas
```

O `projeto.py` segue as seções do artigo:

| Seção | O que faz |
|---|---|
| 1. Coletando os dados | lê o `data.csv` e confere a condição de adequação |
| 2. Explorando e preparando os dados | one-hot das colunas nominais e holdout 80/20 estratificado |
| 3. Treinando o modelo | árvore com `entropy`, `max_depth=6` e `min_samples_split=10`, comparada no k-fold (10) com uma grade de `max_depth` e `class_weight` |
| 4. Resultados no conjunto de teste | `classification_report`, matriz de confusão, `plot_tree`, importância das features |
| 5. Momentos de decisão | ingresso, fim do 1º semestre e fim do 2º semestre |
| 6. Comparação com o KNN | mesmo k-fold, depois árvore x KNN no teste |
| 7. Saída do sistema | lista priorizada com motivo e ação sugerida (`resultados/alertas.csv`) |

Para testar outra configuração, basta mudar os padrões de `criar_arvore` (profundidade e `class_weight`), as listas da grade na seção 3 ou `CAPACIDADE_EQUIPE`, no `projeto.py`.

## Como rodar

Requer Python 3.10 ou mais recente. A execução leva menos de 1 minuto, e as figuras são salvas em `resultados/` em vez de abrir uma janela.

### Windows (PowerShell)

```powershell
cd sistemas-inteligentes-PCD
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python projeto.py
```

Se ainda não tiver o Python, instale pelo [python.org](https://www.python.org/downloads/) ou com `winget install Python.Python.3.12`. Se o PowerShell bloquear o `activate`, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Linux / WSL

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip   # só na primeira vez

cd sistemas-inteligentes-PCD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python projeto.py
```

Dica: o repositório também pode ser acessado pelo WSL em `/mnt/c/repo/sistemas-inteligentes-PCD`, mas clonar dentro do WSL (ex.: `~/repo`) deixa tudo mais rápido.

## Artigo (LaTeX)

`artigo/artigo.tex` usa o template LNCS do professor, com a classe oficial `llncs` na mesma pasta. As figuras vêm direto de `resultados/`, e o apêndice inclui o `projeto.py`.

```bash
python projeto.py     # gera as figuras usadas no artigo
python gerar_pdf.py   # gera artigo/artigo.pdf
```

Para ter o LaTeX:

- **Windows:** instale o [TeX Live](https://tug.org/texlive/), que já traz o `latexmk`.
- **Linux / WSL:** `sudo apt install -y latexmk texlive-latex-extra texlive-fonts-recommended texlive-pictures texlive-lang-portuguese`

Também dá para compilar no Overleaf enviando as pastas `artigo/` e `resultados/` e o `projeto.py`.
