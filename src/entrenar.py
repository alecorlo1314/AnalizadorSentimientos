import time
import numpy as np
import pandas as pd
import mlflow
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import os

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"


# ── Configuración MLflow ───────────────────────────────────────────────────────
EXPERIMENTO = "AnalizadorSentimientos"
mlflow.set_experiment(EXPERIMENTO)


# ── Word2Vec: vectorización de documentos ─────────────────────────────────────
def entrenar_word2vec(tokens_lista: list, vector_size=100, window=5, min_count=2):
    """Entrena Word2Vec sobre el corpus completo."""
    print("Entrenando Word2Vec...")
    modelo = Word2Vec(
        sentences=tokens_lista,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42,
        epochs=10,
    )
    return modelo


def documento_a_vector(tokens: list, modelo_w2v) -> np.ndarray:
    """Promedia los vectores de los tokens presentes en el vocabulario."""
    wv = modelo_w2v.wv
    vectores = [wv[token] for token in tokens if token in wv]
    if not vectores:
        return np.zeros(modelo_w2v.vector_size)
    return np.mean(vectores, axis=0)


def corpus_a_matriz(tokens_lista: list, modelo_w2v) -> np.ndarray:
    """Convierte lista de listas de tokens a matriz de vectores."""
    return np.array([documento_a_vector(t, modelo_w2v) for t in tokens_lista])


# ── Entrenamiento y comparación ───────────────────────────────────────────────
def entrenar_modelos(
    X_train_texto: pd.Series,
    X_train_tokens: list,
    y_train: pd.Series,
):
    """
    Compara 4 combinaciones: (TF-IDF, Word2Vec) x (LogisticRegression, LinearSVC)
    Registra cada run en MLflow y retorna el mejor pipeline.
    """
    resultados = []

    # ── 1. Entrenar Word2Vec una sola vez ──────────────────────────────────────
    modelo_w2v = entrenar_word2vec(X_train_tokens)
    X_w2v = corpus_a_matriz(X_train_tokens, modelo_w2v)

    # ── 2. Configurar TF-IDF ───────────────────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X_train_texto)

    # ── 3. Definir combinaciones ───────────────────────────────────────────────
    combinaciones = [
        {
            "nombre": "TFIDF_LogisticRegression",
            "X": X_tfidf,
            "modelo": LogisticRegression(max_iter=300, C=1.0, random_state=42),
            "params": {
                "vectorizador": "TF-IDF",
                "clasificador": "LogisticRegression",
                "max_features": 20000,
                "C": 1.0,
            },
        },
        {
            "nombre": "TFIDF_LinearSVC",
            "X": X_tfidf,
            "modelo": CalibratedClassifierCV(
                LinearSVC(max_iter=1000, C=1.0, random_state=42)
            ),
            "params": {"vectorizador": "TF-IDF", "clasificador": "LinearSVC", "C": 1.0},
        },
        {
            "nombre": "Word2Vec_LogisticRegression",
            "X": X_w2v,
            "modelo": LogisticRegression(max_iter=300, C=1.0, random_state=42),
            "params": {
                "vectorizador": "Word2Vec",
                "clasificador": "LogisticRegression",
                "vector_size": 100,
                "window": 5,
                "C": 1.0,
            },
        },
        {
            "nombre": "Word2Vec_LinearSVC",
            "X": X_w2v,
            "modelo": CalibratedClassifierCV(
                LinearSVC(max_iter=1000, C=1.0, random_state=42)
            ),
            "params": {
                "vectorizador": "Word2Vec",
                "clasificador": "LinearSVC",
                "vector_size": 100,
                "C": 1.0,
            },
        },
    ]

    # ── 4. Evaluar cada combinación con cross-validation ──────────────────────
    for combo in combinaciones:
        print(f"\nEvaluando: {combo['nombre']}...")
        inicio = time.time()

        scores_f1 = cross_val_score(
            combo["modelo"], combo["X"], y_train, cv=5, scoring="f1", n_jobs=-1
        )
        scores_acc = cross_val_score(
            combo["modelo"], combo["X"], y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        duracion = time.time() - inicio

        f1_mean = scores_f1.mean()
        acc_mean = scores_acc.mean()
        print(
            f"  F1: {f1_mean:.4f} | Accuracy: {acc_mean:.4f} | Tiempo: {duracion:.1f}s"
        )

        # Registrar en MLflow
        with mlflow.start_run(run_name=combo["nombre"]):
            mlflow.log_params(combo["params"])
            mlflow.log_metric("f1_cv", f1_mean)
            mlflow.log_metric("f1_std", scores_f1.std())
            mlflow.log_metric("accuracy_cv", acc_mean)
            mlflow.log_metric("duracion_segundos", duracion)

        resultados.append(
            {
                "nombre": combo["nombre"],
                "f1": f1_mean,
                "accuracy": acc_mean,
                "combo": combo,
            }
        )

    # ── 5. Seleccionar el mejor por F1 ─────────────────────────────────────────
    mejor = max(resultados, key=lambda x: x["f1"])
    print(f"\n✅ Mejor modelo: {mejor['nombre']} (F1={mejor['f1']:.4f})")

    # ── 6. Entrenar el mejor sobre todo el train set ───────────────────────────
    mejor["combo"]["modelo"].fit(mejor["combo"]["X"], y_train)

    return (
        mejor["combo"]["modelo"],
        tfidf if "TFIDF" in mejor["nombre"] else None,
        modelo_w2v if "Word2Vec" in mejor["nombre"] else None,
        mejor["nombre"],
        mejor["f1"],
    )
