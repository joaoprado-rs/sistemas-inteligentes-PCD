"""Criação, k-fold e avaliação dos modelos."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from .config import ARVORE, N_FOLDS, N_JOBS, SEED

# Classes desbalanceadas: além da acurácia, F1 macro e recall de Dropout
METRICAS = {
    'acuracia': 'accuracy',
    'f1_macro': 'f1_macro',
    'recall_dropout': make_scorer(recall_score, labels=['Dropout'], average='macro'),
}
METRICA_DE_ESCOLHA = 'f1_macro'

KFOLD = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def criar_arvore(**ajustes) -> DecisionTreeClassifier:
    """Árvore com config.ARVORE; `ajustes` sobrescreve parâmetros."""
    return DecisionTreeClassifier(**{**ARVORE, **ajustes}, random_state=SEED)


def criar_knn(n_neighbors=5):
    # Normalização dentro do pipeline: o scaler é ajustado só no treino de cada fold
    return make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))


def avaliar_kfold(modelo, x, y) -> dict[str, float]:
    """Média das métricas no k-fold; para a árvore, também a média de regras (folhas)."""
    resultado = cross_validate(modelo, x, y, cv=KFOLD, scoring=METRICAS, n_jobs=N_JOBS,
                               return_estimator=True)
    medias = {nome: resultado[f'test_{nome}'].mean() for nome in METRICAS}
    if isinstance(modelo, DecisionTreeClassifier):
        medias['regras'] = np.mean([arvore.get_n_leaves() for arvore in resultado['estimator']])
    return medias


def avaliar_teste(y_real, y_previsto) -> dict[str, float]:
    return {
        'acuracia': accuracy_score(y_real, y_previsto),
        'f1_macro': f1_score(y_real, y_previsto, average='macro'),
        'recall_dropout': recall_score(y_real, y_previsto, labels=['Dropout'], average='macro'),
    }


def buscar_hiperparametros(criar_modelo, grade, x, y) -> tuple[pd.DataFrame, dict]:
    """Tabela do k-fold para cada combinação e a de maior F1 macro."""
    combinacoes = list(ParameterGrid(grade))
    tabela = pd.DataFrame([{**{k: str(v) for k, v in p.items()},  # str: mostra None, não NaN
                            **avaliar_kfold(criar_modelo(**p), x, y)}
                           for p in combinacoes])
    return tabela, combinacoes[tabela[METRICA_DE_ESCOLHA].idxmax()]


def resumir_regras(arvore: DecisionTreeClassifier) -> dict[str, float]:
    """Número de regras (folhas) e de termos (nós em cada caminho até a folha)."""
    estrutura = arvore.tree_
    profundidade = np.zeros(estrutura.node_count, dtype=int)
    for no in range(estrutura.node_count):  # pai sempre tem id menor que o filho
        for filho in (estrutura.children_left[no], estrutura.children_right[no]):
            if filho != -1:
                profundidade[filho] = profundidade[no] + 1
    termos_por_folha = profundidade[estrutura.children_left == -1]
    return {'regras': len(termos_por_folha), 'termos': int(termos_por_folha.sum()),
            'termos_por_regra': termos_por_folha.mean()}


def avaliar_momentos(modelo, x, y, colunas_ausentes: dict[str, list[str]]) -> pd.DataFrame:
    """K-fold usando só as colunas disponíveis em cada momento."""
    linhas = []
    for momento, ausentes in colunas_ausentes.items():
        x_momento = x.drop(columns=ausentes)
        linhas.append({'momento': momento, 'features': x_momento.shape[1],
                       **avaliar_kfold(modelo, x_momento, y)})
    return pd.DataFrame(linhas)
