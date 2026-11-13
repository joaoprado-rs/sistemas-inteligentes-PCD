"""Carga e preparação dos dados."""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ARQUIVO_DADOS, PROPORCAO_TESTE, SEED

COLUNA_ALVO = 'Target'

# Categóricas codificadas como inteiros (ex.: Course = 9500); precisam de one-hot explícito
COLUNAS_NOMINAIS = [
    'Marital status', 'Application mode', 'Course', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
]


@dataclass
class Dados:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    colunas_one_hot: frozenset[str]
    colunas_binarias: frozenset[str]


def carregar_dados(caminho=ARQUIVO_DADOS) -> pd.DataFrame:
    dataset = pd.read_csv(caminho, sep=';', encoding='utf-8-sig')  # arquivo com BOM
    dataset.columns = dataset.columns.str.strip()  # alguns nomes terminam com tab
    return dataset


def contar_conflitos(dataset: pd.DataFrame) -> int:
    """Quantos grupos de alunos têm as mesmas features e desfechos diferentes."""
    features = [c for c in dataset.columns if c != COLUNA_ALVO]
    return int((dataset.groupby(features)[COLUNA_ALVO].nunique() > 1).sum())


def preparar_dados(dataset: pd.DataFrame) -> Dados:
    y = dataset[COLUNA_ALVO]
    x = dataset.drop(columns=[COLUNA_ALVO])

    colunas_binarias = frozenset(c for c in x.columns
                                 if c not in COLUNAS_NOMINAIS and set(x[c].unique()) <= {0, 1})
    x = pd.get_dummies(x, columns=COLUNAS_NOMINAIS)
    colunas_one_hot = frozenset(c for c in x.columns if c.rsplit('_', 1)[0] in COLUNAS_NOMINAIS)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=PROPORCAO_TESTE, random_state=SEED, stratify=y
    )
    return Dados(x_train, x_test, y_train, y_test, colunas_one_hot, colunas_binarias)


def colunas_ausentes_por_momento(colunas) -> dict[str, list[str]]:
    """Colunas que ainda não existem em cada momento de decisão."""
    colunas_1sem = [c for c in colunas if c.startswith('Curricular units 1st sem')]
    colunas_2sem = [c for c in colunas if c.startswith('Curricular units 2nd sem')]
    return {
        'Ingresso (matrícula)': colunas_1sem + colunas_2sem,
        'Fim do 1º semestre': colunas_2sem,
        'Fim do 2º semestre': [],
    }
