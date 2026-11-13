"""Figuras salvas em arquivo."""
import matplotlib
matplotlib.use('Agg')  # sem janela (funciona no WSL)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


def salvar_matriz_confusao(y_real, y_previsto, classes, caminho):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_real, y_previsto, labels=classes, cmap='Blues', ax=ax)
    ax.set(title='Matriz de confusão - árvore de decisão (teste)',
           xlabel='Classe prevista', ylabel='Classe real')
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def salvar_arvore(arvore, caminho, profundidade=3):
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(arvore, feature_names=list(arvore.feature_names_in_), class_names=list(arvore.classes_),
              filled=True, max_depth=profundidade, fontsize=8, ax=ax)
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close(fig)


def salvar_importancias(arvore, caminho, n=10):
    importancias = pd.Series(arvore.feature_importances_, index=arvore.feature_names_in_).nlargest(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.sort_values().plot.barh(ax=ax)
    ax.set_title(f'{n} features mais importantes da árvore')
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
