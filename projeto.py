"""
Projeto de Conclusão de Disciplina - Sistemas Inteligentes (UFABC, 2026.Q3)

Sistema de alerta precoce de evasão: uma árvore de decisão estima o desfecho de cada
aluno ativo (Dropout, Enrolled ou Graduate) e o sistema gera uma lista priorizada
com o motivo do alerta e uma ação sugerida.

Uso: python projeto.py   (resultados em resultados/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # salva as figuras sem abrir janela (funciona no WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             get_scorer, make_scorer, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.stdout.reconfigure(encoding='utf-8')  # acentos e ≤ também no Windows

PASTA = Path(__file__).resolve().parent
PASTA_RESULTADOS = PASTA / 'resultados'
PASTA_RESULTADOS.mkdir(exist_ok=True)

SEED = 42
CAPACIDADE_EQUIPE = 20  # alunos atendidos por período


def imprimir_tabela(tabela):
    print(tabela.round(3).to_string(index=False))


### 1. Coletando os dados

dataset = pd.read_csv(PASTA / 'data.csv', sep=';', encoding='utf-8-sig')  # arquivo com BOM
dataset.columns = dataset.columns.str.strip()  # alguns nomes terminam com tab

print('=== 1. Dados ===')
print(f'{dataset.shape[0]} alunos, {dataset.shape[1] - 1} features + coluna alvo')
print(dataset['Target'].value_counts().to_string(), '\n')

# Condição de adequação: alunos com as mesmas features não podem ter desfechos diferentes
features = list(dataset.columns.drop('Target'))
conflitos = (dataset.groupby(features)['Target'].nunique() > 1).sum()
print(f'Condição de adequação: {conflitos} grupos de alunos com '
      'as mesmas features e desfechos diferentes\n')


### 2. Explorando e preparando os dados

variavel_classificadora = dataset['Target']
dataset = dataset.drop(columns=['Target'])
colunas_binarias = list(dataset.columns[dataset.isin([0, 1]).all()])  # usadas no motivo (seção 7)

# Categóricas vêm como inteiros (ex.: Course = 9500): sem columns=, o get_dummies não as converte
COLUNAS_NOMINAIS = [
    'Marital status', 'Application mode', 'Course', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
]
dataset = pd.get_dummies(dataset, columns=COLUNAS_NOMINAIS)
print(f'Após o one-hot encoding: {dataset.shape[1]} colunas\n')

# Holdout 80/20 estratificado; o teste também faz o papel dos alunos ativos (seção 7)
x_train, x_test, y_train, y_test = train_test_split(
    dataset, variavel_classificadora, test_size=0.2, random_state=SEED,
    stratify=variavel_classificadora
)


### 3. Treinando o modelo

def criar_arvore(max_depth=6, class_weight=None):
    return DecisionTreeClassifier(
        criterion='entropy',        # ganho de informação
        max_depth=max_depth,        # evita overfitting
        min_samples_split=10,       # mínimo de alunos para dividir um nó
        class_weight=class_weight,  # 'balanced' dá mais peso às classes menores
        random_state=SEED,
    )


# Classes desbalanceadas: além da acurácia, F1 macro e recall de Dropout
METRICAS = {
    'acuracia': 'accuracy',
    'f1_macro': 'f1_macro',
    'recall_dropout': make_scorer(recall_score, labels=['Dropout'], average='macro'),
}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)


def avaliar_kfold(modelo, x, y):
    """Média das métricas nos 10 folds; para a árvore, também a média de regras (folhas)."""
    resultado = cross_validate(modelo, x, y, cv=kfold, scoring=METRICAS, n_jobs=-1,  # em paralelo
                               return_estimator=True)
    medias = {nome: resultado[f'test_{nome}'].mean() for nome in METRICAS}
    if isinstance(modelo, DecisionTreeClassifier):
        medias['regras'] = np.mean([arvore.get_n_leaves() for arvore in resultado['estimator']])
    return medias


def avaliar_teste(modelo):
    return {nome: get_scorer(metrica)(modelo, x_test, y_test) for nome, metrica in METRICAS.items()}


# A configuração adotada (padrão de criar_arvore) comparada com outras profundidades
print('=== 3. Árvore de decisão: k-fold (10) no conjunto de treino ===')
grade = pd.DataFrame([
    {'class_weight': str(class_weight), 'max_depth': str(max_depth),  # str: mostra None, não NaN
     **avaliar_kfold(criar_arvore(max_depth, class_weight), x_train, y_train)}
    for class_weight in [None, 'balanced'] for max_depth in [3, 4, 5, 6, 8, 10, None]
])
imprimir_tabela(grade)
melhor = grade.loc[grade['f1_macro'].idxmax()]
print(f'\nMaior F1 macro: class_weight={melhor["class_weight"]}, max_depth={melhor["max_depth"]} '
      f'({melhor["f1_macro"]:.3f}, {melhor["regras"]:.0f} regras). Adotada: max_depth=6\n')


### 4. Resultados no conjunto de teste

model = criar_arvore()
model.fit(x_train, y_train)
pred = model.predict(x_test)
classes = list(model.classes_)

print('=== 4. Árvore de decisão: resultado no teste (holdout 20%) ===')
print(classification_report(y_test, pred, digits=3))
print('Matriz de confusão (linhas = real, colunas = previsto):', classes)
print(confusion_matrix(y_test, pred, labels=classes), '\n')

# Cada folha é uma regra SE ... ENTÃO; os nós no caminho até ela são os termos
folhas = model.tree_.children_left == -1
termos = model.tree_.compute_node_depths()[folhas] - 1  # a raiz tem profundidade 1
print(f'A árvore resume os {len(x_train)} alunos de treino em {folhas.sum()} regras, '
      f'com {termos.sum()} termos ({termos.mean():.1f} por regra em média)\n')

ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=classes, cmap='Blues')
plt.title('Matriz de confusão - árvore de decisão (teste)')
plt.xlabel('Classe prevista')
plt.ylabel('Classe real')
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / 'matriz_confusao.png', dpi=150)
plt.close()

plt.figure(figsize=(24, 10))
plot_tree(model, feature_names=list(x_train.columns), class_names=classes,
          filled=True, max_depth=3, fontsize=8)  # só os primeiros níveis
plt.savefig(PASTA_RESULTADOS / 'arvore.png', dpi=150, bbox_inches='tight')
plt.close()

importancias = pd.Series(model.feature_importances_, index=x_train.columns).nlargest(10)
importancias.sort_values().plot.barh(figsize=(8, 5))
plt.title('10 features mais importantes da árvore')
plt.tight_layout()
plt.savefig(PASTA_RESULTADOS / 'importancia_features.png', dpi=150)
plt.close()


### 5. Momentos de decisão

# Colunas que ainda não existem em cada momento
colunas_1sem = [c for c in x_train.columns if c.startswith('Curricular units 1st sem')]
colunas_2sem = [c for c in x_train.columns if c.startswith('Curricular units 2nd sem')]
MOMENTOS = {
    'Ingresso (matrícula)': colunas_1sem + colunas_2sem,
    'Fim do 1º semestre': colunas_2sem,
    'Fim do 2º semestre': [],
}

print('=== 5. Momentos de decisão (mesma configuração, k-fold no treino) ===')
imprimir_tabela(pd.DataFrame([
    {'momento': momento, 'features': x_train.shape[1] - len(ausentes),
     **avaliar_kfold(criar_arvore(), x_train.drop(columns=ausentes), y_train)}
    for momento, ausentes in MOMENTOS.items()
]))
print()


### 6. Comparação com o KNN

def criar_knn(k):
    # Normalização dentro do pipeline: o scaler é ajustado só no treino de cada fold
    return make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k))


print('=== 6. KNN: k-fold (10) no conjunto de treino ===')
grade_knn = pd.DataFrame([{'n_neighbors': k, **avaliar_kfold(criar_knn(k), x_train, y_train)}
                          for k in [5, 11, 21, 31, 51]])
imprimir_tabela(grade_knn)
k = grade_knn.loc[grade_knn['f1_macro'].idxmax(), 'n_neighbors']
knn = criar_knn(k).fit(x_train, y_train)

print(f'\n=== 6. Árvore x KNN (k = {k}) no teste ===')
imprimir_tabela(pd.DataFrame([
    {'modelo': 'Árvore de decisão', **avaliar_teste(model)},
    {'modelo': f'KNN (k = {k})', **avaliar_teste(knn)},
]))
print()


### 7. Saída do sistema: lista priorizada de alertas

def explicar_caminho(arvore, x):
    """Condições do caminho de cada aluno até a folha: {coluna: (inferior, superior)}."""
    estrutura = arvore.tree_
    caminhos = arvore.decision_path(x)
    explicacoes = []
    for i in range(x.shape[0]):
        nos = caminhos.indices[caminhos.indptr[i]:caminhos.indptr[i + 1]]  # raiz -> folha
        limites = {}
        for no, filho in zip(nos, nos[1:]):
            coluna = x.columns[estrutura.feature[no]]
            limiar = estrutura.threshold[no]
            inferior, superior = limites.get(coluna, (None, None))
            if filho == estrutura.children_left[no]:  # esquerda: coluna <= limiar
                superior = limiar if superior is None else min(superior, limiar)
            else:
                inferior = limiar if inferior is None else max(inferior, limiar)
            limites[coluna] = (inferior, superior)
        explicacoes.append(limites)
    return explicacoes


def descrever_condicao(coluna, inferior, superior):
    nome, _, codigo = coluna.rpartition('_')
    if nome in COLUNAS_NOMINAIS:  # one-hot: Course_9500 -> "Course = 9500"
        return f'{nome} = {codigo}' if inferior is not None else f'{nome} ≠ {codigo}'
    if coluna in colunas_binarias:
        return f'{coluna} = {1 if inferior is not None else 0}'
    if inferior is not None and superior is not None:
        return f'{inferior:.4g} < {coluna} ≤ {superior:.4g}'
    if superior is not None:
        return f'{coluna} ≤ {superior:.4g}'
    return f'{coluna} > {inferior:.4g}'


def sugerir_acao(nivel, limites):
    """Regras fixas da equipe pedagógica (não aprendidas pelo modelo)."""
    if nivel == 'Sem alerta':
        return '-'
    acoes = []
    mensalidade_atrasada = limites.get('Tuition fees up to date', (None, None))[1] is not None
    devedor = limites.get('Debtor', (None, None))[0] is not None
    if mensalidade_atrasada or devedor:
        acoes.append('encaminhar à assistência estudantil (situação financeira)')
    baixo_desempenho = any(
        c.startswith('Curricular units') and ('approved' in c or 'grade' in c) and sup is not None
        for c, (_, sup) in limites.items()
    )
    if baixo_desempenho:
        acoes.append('oferecer monitoria/tutoria acadêmica')
    return '; '.join(acoes) if acoes else 'conversa com a coordenação de curso'


# Simulação: os alunos do teste fazem o papel dos alunos ativos
probabilidades = model.predict_proba(x_test)
caminhos = explicar_caminho(model, x_test)

alertas = pd.DataFrame({
    'aluno': x_test.index,  # linha no data.csv
    'prob_evasao': probabilidades[:, classes.index('Dropout')],
    'prob_retencao': probabilidades[:, classes.index('Enrolled')],
    'previsao': pred,
})
alertas['nivel'] = np.select(
    [alertas['previsao'] == 'Dropout', alertas['previsao'] == 'Enrolled'],
    ['Alto (risco de evasão)', 'Médio (risco de retenção)'], default='Sem alerta'
)
alertas['motivo'] = [' e '.join(descrever_condicao(c, inf, sup) for c, (inf, sup) in limites.items())
                     for limites in caminhos]
alertas['acao_sugerida'] = [sugerir_acao(n, l) for n, l in zip(alertas['nivel'], caminhos)]
alertas['desfecho_real'] = y_test.values  # só existe na simulação

alertas = alertas.sort_values(['prob_evasao', 'prob_retencao'], ascending=False).reset_index(drop=True)
alertas.insert(0, 'prioridade', alertas.index + 1)
alertas.to_csv(PASTA_RESULTADOS / 'alertas.csv', sep=';', index=False, encoding='utf-8-sig')

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
print('\nLista completa salva em resultados/alertas.csv')
