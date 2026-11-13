"""Lista priorizada de alertas, com motivo e ação sugerida."""
import numpy as np
import pandas as pd

NIVEL_ALTO = 'Alto (risco de evasão)'
NIVEL_MEDIO = 'Médio (risco de retenção)'
SEM_ALERTA = 'Sem alerta'


def explicar_caminho(arvore, x: pd.DataFrame) -> list[dict[str, tuple]]:
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


def descrever_condicao(coluna, inferior, superior, colunas_one_hot, colunas_binarias) -> str:
    if coluna in colunas_one_hot:  # Course_9500 -> "Course = 9500"
        nome, codigo = coluna.rsplit('_', 1)
        return f'{nome} = {codigo}' if inferior is not None else f'{nome} ≠ {codigo}'
    if coluna in colunas_binarias:
        return f'{coluna} = {1 if inferior is not None else 0}'
    if inferior is not None and superior is not None:
        return f'{inferior:.4g} < {coluna} ≤ {superior:.4g}'
    if superior is not None:
        return f'{coluna} ≤ {superior:.4g}'
    return f'{coluna} > {inferior:.4g}'


def sugerir_acao(nivel, limites) -> str:
    """Regras fixas da equipe pedagógica (não aprendidas pelo modelo)."""
    if nivel == SEM_ALERTA:
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


def gerar_alertas(arvore, x, colunas_one_hot, colunas_binarias, desfecho_real=None) -> pd.DataFrame:
    """Alertas ordenados pela probabilidade de evasão."""
    probabilidades = arvore.predict_proba(x)
    classes = list(arvore.classes_)
    caminhos = explicar_caminho(arvore, x)

    alertas = pd.DataFrame({
        'aluno': x.index,  # linha no data.csv
        'prob_evasao': probabilidades[:, classes.index('Dropout')],
        'prob_retencao': probabilidades[:, classes.index('Enrolled')],
        'previsao': arvore.predict(x),
    })
    alertas['nivel'] = np.select(
        [alertas['previsao'] == 'Dropout', alertas['previsao'] == 'Enrolled'],
        [NIVEL_ALTO, NIVEL_MEDIO], default=SEM_ALERTA
    )
    alertas['motivo'] = [
        ' e '.join(descrever_condicao(c, inf, sup, colunas_one_hot, colunas_binarias)
                   for c, (inf, sup) in limites.items())
        for limites in caminhos
    ]
    alertas['acao_sugerida'] = [sugerir_acao(n, l) for n, l in zip(alertas['nivel'], caminhos)]
    if desfecho_real is not None:  # só na simulação
        alertas['desfecho_real'] = np.asarray(desfecho_real)

    alertas = alertas.sort_values(['prob_evasao', 'prob_retencao'], ascending=False).reset_index(drop=True)
    alertas.insert(0, 'prioridade', alertas.index + 1)
    return alertas
