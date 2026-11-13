# sistemas-inteligentes-PCD
Projeto de conclusão de disciplina da matéria Sistemas Inteligentes (UFABC, 2026.Q3).

**Sistema de alerta precoce de evasão.** Ao fim de cada período letivo, uma árvore de decisão estima o desfecho de cada aluno ativo (evasão, retenção ou formatura), e o sistema entrega à coordenação uma lista priorizada com o motivo do alerta e uma ação sugerida.

Dataset: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (UCI, CC BY 4.0), em `data.csv`.

## Estrutura

```
data.csv              dataset (separador ';')
projeto.py            ponto de entrada: executa as seções do artigo em ordem e imprime os resultados
evasao/
  config.py           parâmetros do experimento: configuração da árvore, semente, grades
  dados.py            carga, one-hot das colunas nominais e holdout
  modelo.py           árvore e KNN: criação, k-fold, escolha de hiperparâmetros, avaliação no teste
  graficos.py         matriz de confusão, desenho da árvore, importância das features
  alertas.py          lista priorizada de alertas: motivo (caminho na árvore) e ação sugerida
gerar_pdf.py          compila o artigo LaTeX em artigo/artigo.pdf
artigo/               artigo em LaTeX (template LNCS do professor)
requirements.txt      dependências
resultados/           gerado ao rodar: figuras e lista de alertas
```

O `projeto.py` segue as seções do artigo:

| Seção | O que faz | Onde está a lógica |
|---|---|---|
| 1. Coleta dos dados | lê o `data.csv` | `evasao/dados.py` |
| 2. Preparação | one-hot das colunas nominais e holdout 80/20 estratificado | `evasao/dados.py` |
| 3. Configuração da árvore | `entropy`, `max_depth=6`, `min_samples_split=10`, comparada no k-fold (10) com uma grade de `max_depth` e `class_weight` | `evasao/config.py`, `evasao/modelo.py` |
| 4. Resultados no teste | `classification_report`, matriz de confusão, `plot_tree`, importância das features | `evasao/modelo.py`, `evasao/graficos.py` |
| 5. Momentos de decisão | ingresso, fim do 1º semestre e fim do 2º semestre | `evasao/dados.py`, `evasao/modelo.py` |
| 6. Comparação com o KNN | mesmo k-fold, depois árvore x KNN no teste | `evasao/modelo.py` |
| 7. Saída do sistema | lista priorizada com motivo e ação sugerida (`resultados/alertas.csv`) | `evasao/alertas.py` |

Para testar outra configuração (outra profundidade da árvore, outra grade, outro `k` do KNN, outra capacidade da equipe), basta mudar `evasao/config.py`.

## Como rodar (WSL / Linux)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip   # só na primeira vez

cd sistemas-inteligentes-PCD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python projeto.py
```

Requer Python 3.10 ou mais recente. A execução leva menos de 1 minuto. As figuras são salvas em `resultados/` em vez de abrir uma janela.

Dica: o repositório também pode ser acessado pelo WSL em `/mnt/c/repo/sistemas-inteligentes-PCD`, mas clonar dentro do WSL (ex.: `~/repo`) deixa tudo mais rápido.

## Artigo (LaTeX)

`artigo/artigo.tex` é o template Word do professor (LNCS da Springer) convertido para LaTeX, com a classe oficial `llncs` na mesma pasta. As figuras de `resultados/` podem ser incluídas direto pelo nome, ex.: `\includegraphics[width=\textwidth]{matriz_confusao.png}`.

```bash
sudo apt install -y latexmk texlive-latex-extra texlive-fonts-recommended texlive-pictures texlive-lang-portuguese   # só na primeira vez

python projeto.py     # gera as figuras usadas no artigo
python gerar_pdf.py   # gera artigo/artigo.pdf
```

O template está em inglês, como o original. Para escrever em português, descomente o bloco indicado no início do `artigo.tex`. Também dá para compilar no Overleaf enviando a pasta `artigo/` (e `resultados/`, se usar as figuras).
