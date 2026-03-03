import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lime.lime_text import LimeTextExplainer

os.makedirs("Resultados", exist_ok=True)

CLASES = ["Negativo", "Positivo"]
explainer = LimeTextExplainer(class_names=CLASES)


def _predict_fn(tfidf, modelo):
    """Retorna una función de predicción compatible con LIME."""
    def predict(textos):
        X = tfidf.transform(textos)
        return modelo.predict_proba(X)
    return predict


def explicar_prediccion(
    texto_limpio: str,
    tfidf,
    modelo,
    num_features: int = 10,
    num_samples: int = 500,
) -> tuple:
    """
    Genera explicación LIME para un texto individual.
    Retorna (explicacion, lista de (palabra, peso)).
    """
    predict_fn = _predict_fn(tfidf, modelo)
    exp = explainer.explain_instance(
        texto_limpio,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )
    return exp, exp.as_list()


def graficar_explicacion(palabras_pesos: list, titulo: str = "Explicación LIME") -> plt.Figure:
    """
    Genera gráfica de barras horizontal con los pesos LIME.
    Verde = empuja hacia Positivo, Rojo = empuja hacia Negativo.
    """
    if not palabras_pesos:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return fig

    palabras, pesos = zip(*palabras_pesos)
    colores = ["#4caf7d" if p > 0 else "#e05c5c" for p in pesos]

    # Ordenar por peso absoluto
    orden = sorted(range(len(pesos)), key=lambda i: pesos[i])
    palabras = [palabras[i] for i in orden]
    pesos = [pesos[i] for i in orden]
    colores = [colores[i] for i in orden]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.barh(palabras, pesos, color=colores, edgecolor="none", height=0.6)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Peso LIME", color="#aaa", fontsize=9)
    ax.tick_params(colors="#ccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(titulo, color="white", fontsize=11, pad=10)

    leyenda = [
        mpatches.Patch(facecolor="#4caf7d", label="→ Positivo"),
        mpatches.Patch(facecolor="#e05c5c", label="→ Negativo"),
    ]
    ax.legend(handles=leyenda, loc="lower right",
              facecolor="#1a1d27", edgecolor="#333", labelcolor="#ccc", fontsize=8)

    plt.tight_layout()
    return fig


def guardar_explicacion_global(tfidf, modelo, X_test_texto, y_test, n_muestras: int = 5):
    """
    Genera y guarda explicaciones LIME para n_muestras del test set
    (una positiva y una negativa como mínimo) en Resultados/.
    """
    predict_fn = _predict_fn(tfidf, modelo)
    indices_pos = [i for i, y in enumerate(y_test) if y == 1][:3]
    indices_neg = [i for i, y in enumerate(y_test) if y == 0][:2]

    for idx, etiqueta in zip(indices_pos + indices_neg,
                              ["positivo"] * 3 + ["negativo"] * 2):
        texto = X_test_texto.iloc[idx]
        exp = explainer.explain_instance(texto, predict_fn,
                                         num_features=10, num_samples=300)
        fig = graficar_explicacion(exp.as_list(), titulo=f"LIME — ejemplo {etiqueta}")
        ruta = f"Resultados/lime_{etiqueta}_{idx}.png"
        fig.savefig(ruta, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Explicación guardada: {ruta}")