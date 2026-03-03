import os
import pandas as pd
from src.datos import cargar_datos, separar_features, dividir_datos
from src.preprocesar import preprocesar_serie, tokens_a_texto
from src.entrenar import entrenar_modelos, corpus_a_matriz
from src.evaluar import evaluar_modelo, generar_reporte
from src.guardar import guardar_modelo, guardar_tfidf, guardar_word2vec
import os

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

os.makedirs("Resultados", exist_ok=True)

# ── 1. Cargar datos ────────────────────────────────────────────────────────────
print("=" * 50)
print("1. Cargando datos...")
df = cargar_datos("Datos/IMDB Dataset SPANISH.csv")
X, y = separar_features(df)
X_train, X_test, y_train, y_test = dividir_datos(X, y)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 2. Preprocesar ─────────────────────────────────────────────────────────────
print("\n2. Preprocesando texto...")
X_train_tokens = preprocesar_serie(X_train).tolist()
X_test_tokens = preprocesar_serie(X_test).tolist()

X_train_texto = pd.Series([" ".join(t) for t in X_train_tokens])
X_test_texto = pd.Series([" ".join(t) for t in X_test_tokens])

# ── 3. Entrenar y comparar modelos ─────────────────────────────────────────────
print("\n3. Entrenando y comparando modelos con MLflow...")
modelo, tfidf, modelo_w2v, nombre_modelo, f1_cv = entrenar_modelos(
    X_train_texto, X_train_tokens, y_train
)

# ── 4. Preparar X_test según el mejor vectorizador ────────────────────────────
print("\n4. Preparando datos de test...")
if tfidf is not None:
    X_test_final = tfidf.transform(X_test_texto)
else:
    X_test_final = corpus_a_matriz(X_test_tokens, modelo_w2v)

# ── 5. Evaluar ─────────────────────────────────────────────────────────────────
print("\n5. Evaluando en test set...")
f1, precision, recall, accuracy = evaluar_modelo(modelo, X_test_final, y_test)

# ── 6. Generar reporte ─────────────────────────────────────────────────────────
print("\n6. Generando reporte...")
generar_reporte(f1, precision, recall, accuracy, nombre_modelo)

# ── 7. Guardar modelo y vectorizador ───────────────────────────────────────────
print("\n7. Guardando modelo...")
guardar_modelo(modelo, nombre_modelo)
if tfidf is not None:
    guardar_tfidf(tfidf)
if modelo_w2v is not None:
    guardar_word2vec(modelo_w2v)

print("\n" + "=" * 50)
print(f"✅ Pipeline completado — Mejor modelo: {nombre_modelo}")
print(f"   F1 CV: {f1_cv:.4f} | F1 Test: {f1:.4f} | Accuracy: {accuracy:.4f}")
print("=" * 50)

# ── 8. Generar explicaciones LIME (solo si el mejor modelo usa TF-IDF) ─────────
if tfidf is not None:
    print("\n8. Generando explicaciones LIME...")
    from src.explicar import guardar_explicacion_global

    guardar_explicacion_global(
        tfidf, modelo, X_test_texto, y_test.reset_index(drop=True)
    )
