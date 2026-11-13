"""
Projeto de Conclusão de Disciplina - Sistemas Inteligentes (UFABC, 2026.Q3)

Sistema de alerta precoce de evasão: uma árvore de decisão estima o desfecho de cada
aluno ativo (Dropout, Enrolled ou Graduate) e o sistema gera uma lista priorizada
com o motivo do alerta e uma ação sugerida.

Uso: python projeto.py   (resultados em resultados/)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from evasao.alertas import gerar_alertas
from evasao.config import (ARVORE, CAPACIDADE_EQUIPE, GRADE_ARVORE, GRADE_KNN, N_FOLDS,
                           PASTA_RESULTADOS, PROPORCAO_TESTE, RAIZ)
from evasao.dados import (COLUNA_ALVO, Dados, carregar_dados, colunas_ausentes_por_momento,
                          contar_conflitos, preparar_dados)
from evasao.graficos import salvar_arvore, salvar_importancias, salvar_matriz_confusao
from evasao.modelo import (avaliar_momentos, avaliar_teste, buscar_hiperparametros,
                           criar_arvore, criar_knn, resumir_regras)


def imprimir_tabela(tabela: pd.DataFrame):
    print(tabela.round(3).to_string(index=False))


def linha_da_grade(tabela: pd.DataFrame, parametros: dict) -> pd.Series:
    mascara = np.logical_and.reduce([tabela[k] == str(v) for k, v in parametros.items()])
    return tabela[mascara].iloc[0]


### 1 e 2. Coleta e preparação dos dados

def secao_dados() -> Dados:
    dataset = carregar_dados()
    print('=== 1. Dados ===')
    print(f'{dataset.shape[0]} alunos, {dataset.shape[1] - 1} features + coluna alvo')
    print(dataset[COLUNA_ALVO].value_counts().to_string(), '\n')
    print(f'Condição de adequação: {contar_conflitos(dataset)} grupos de alunos com '
          'as mesmas features e desfechos diferentes\n')

    dados = preparar_dados(dataset)
    print(f'Após o one-hot encoding: {dados.x_train.shape[1]} colunas\n')
    return dados


### 3. Configuração da árvore

def secao_configuracao(dados: Dados):
    print(f'=== 3. Árvore de decisão: k-fold ({N_FOLDS}) no conjunto de treino ===')
    tabela, melhor = buscar_hiperparametros(criar_arvore, GRADE_ARVORE, dados.x_train, dados.y_train)
    imprimir_tabela(tabela)

    usada = linha_da_grade(tabela, {k: ARVORE[k] for k in GRADE_ARVORE})
    maior_f1 = linha_da_grade(tabela, melhor)
    print(f'\nConfiguração usada: {ARVORE}')
    print(f'  k-fold: F1 macro {usada["f1_macro"]:.3f} com {usada["regras"]:.0f} regras em média')
    print(f'  Maior F1 macro da grade: {melhor} -> {maior_f1["f1_macro"]:.3f} '
          f'com {maior_f1["regras"]:.0f} regras\n')


### 4. Resultados no conjunto de teste

def secao_teste(dados: Dados):
    arvore = criar_arvore().fit(dados.x_train, dados.y_train)
    pred = arvore.predict(dados.x_test)
    classes = list(arvore.classes_)

    print(f'=== 4. Árvore de decisão: resultado no teste (holdout {PROPORCAO_TESTE:.0%}) ===')
    print(classification_report(dados.y_test, pred, digits=3))
    print('Matriz de confusão (linhas = real, colunas = previsto):', classes)
    print(confusion_matrix(dados.y_test, pred, labels=classes), '\n')

    regras = resumir_regras(arvore)
    print(f'A árvore resume os {len(dados.x_train)} alunos de treino em {regras["regras"]} regras, '
          f'com {regras["termos"]} termos ({regras["termos_por_regra"]:.1f} por regra em média)\n')

    salvar_matriz_confusao(dados.y_test, pred, classes, PASTA_RESULTADOS / 'matriz_confusao.png')
    salvar_arvore(arvore, PASTA_RESULTADOS / 'arvore.png')
    salvar_importancias(arvore, PASTA_RESULTADOS / 'importancia_features.png')
    return arvore, pred


### 5. Momentos de decisão

def secao_momentos(dados: Dados):
    print('=== 5. Momentos de decisão (mesma configuração, k-fold no treino) ===')
    ausentes = colunas_ausentes_por_momento(dados.x_train.columns)
    imprimir_tabela(avaliar_momentos(criar_arvore(), dados.x_train, dados.y_train, ausentes))
    print()


### 6. Comparação com o KNN

def secao_knn(dados: Dados, pred_arvore):
    print(f'=== 6. KNN: k-fold ({N_FOLDS}) no conjunto de treino ===')
    tabela, parametros = buscar_hiperparametros(criar_knn, GRADE_KNN, dados.x_train, dados.y_train)
    imprimir_tabela(tabela)

    k = parametros['n_neighbors']
    pred_knn = criar_knn(**parametros).fit(dados.x_train, dados.y_train).predict(dados.x_test)
    print(f'\n=== 6. Árvore x KNN (k = {k}) no teste ===')
    imprimir_tabela(pd.DataFrame([
        {'modelo': 'Árvore de decisão', **avaliar_teste(dados.y_test, pred_arvore)},
        {'modelo': f'KNN (k = {k})', **avaliar_teste(dados.y_test, pred_knn)},
    ]))
    print()


### 7. Saída do sistema: lista priorizada de alertas

def secao_alertas(dados: Dados, arvore):
    # Simulação: os alunos do teste fazem o papel dos alunos ativos
    alertas = gerar_alertas(arvore, dados.x_test, dados.colunas_one_hot, dados.colunas_binarias,
                            desfecho_real=dados.y_test)
    caminho = PASTA_RESULTADOS / 'alertas.csv'
    alertas.to_csv(caminho, sep=';', index=False, encoding='utf-8-sig')

    print('=== 7. Saída do sistema (simulação com os alunos do teste) ===')
    print(alertas['nivel'].value_counts().to_string())
    priorizados = alertas.head(CAPACIDADE_EQUIPE)
    acertos = (priorizados['desfecho_real'] == 'Dropout').sum()
    print(f'\nDos {CAPACIDADE_EQUIPE} alunos que a equipe atenderia primeiro, {acertos} de fato evadiram.')
    print(priorizados[['prioridade', 'aluno', 'prob_evasao', 'nivel', 'acao_sugerida']]
          .round(2).to_string(index=False))
    print('\nMotivo dos 3 primeiros alertas:')
    for _, linha in alertas.head(3).iterrows():
        print(f"  Aluno {linha['aluno']}: {linha['motivo']}")
    print(f'\nLista completa salva em {caminho.relative_to(RAIZ)}')


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    dados = secao_dados()
    secao_configuracao(dados)
    arvore, pred = secao_teste(dados)
    secao_momentos(dados)
    secao_knn(dados, pred)
    secao_alertas(dados, arvore)


if __name__ == '__main__':
    main()
