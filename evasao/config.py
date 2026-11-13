"""Parâmetros do experimento."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_DADOS = RAIZ / 'data.csv'
PASTA_RESULTADOS = RAIZ / 'resultados'

SEED = 42
PROPORCAO_TESTE = 0.2   # mesma divisão dos autores do dataset
N_FOLDS = 10
N_JOBS = -1             # k-fold em paralelo, usando todos os núcleos
CAPACIDADE_EQUIPE = 20  # alunos atendidos por período

ARVORE = {
    'criterion': 'entropy',   # ganho de informação
    'max_depth': 6,           # evita overfitting
    'min_samples_split': 10,  # mínimo de alunos para dividir um nó
    'class_weight': None,
}

# Grades comparadas no k-fold
GRADE_ARVORE = {
    'class_weight': [None, 'balanced'],
    'max_depth': [3, 4, 5, 6, 8, 10, None],
}
GRADE_KNN = {'n_neighbors': [5, 11, 21, 31, 51]}
