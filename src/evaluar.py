import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

os.makedirs("Resultados", exist_ok=True)


def evaluar_modelo(modelo, X_test, y_test, umbral=0.5):
    """Evalúa el modelo y genera todas las gráficas."""
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n=== Métricas con umbral={umbral} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))

    # Guardar métricas en texto
    with open("Resultados/metricas.txt", "w", encoding="utf-8") as f:
        f.write(f"Umbral: {umbral}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    clases = ["Negativo", "Positivo"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=clases,
        yticklabels=clases,
        title="Matriz de Confusión",
        ylabel="Real",
        xlabel="Predicho",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("Resultados/confusion_matrix.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("Resultados/roc_curve.png")
    plt.close()

    print("Gráficas guardadas en Resultados/")
    return f1, precision, recall, accuracy


def generar_reporte(f1, precision, recall, accuracy, nombre_modelo: str):
    """Genera reporte Markdown para CML."""
    with open("Resultados/reporte.md", "w", encoding="utf-8") as f:
        f.write(f"## Reporte del Modelo — {nombre_modelo}\n\n")
        f.write("| Métrica | Valor |\n")
        f.write("|---|---|\n")
        f.write(f"| Accuracy  | {accuracy:.4f} |\n")
        f.write(f"| F1-Score  | {f1:.4f} |\n")
        f.write(f"| Precision | {precision:.4f} |\n")
        f.write(f"| Recall    | {recall:.4f} |\n")
        f.write("\n### Matriz de Confusión\n\n")
        f.write("![Matriz de Confusión](Resultados/confusion_matrix.png)\n\n")
        f.write("### Curva ROC\n\n")
        f.write("![Curva ROC](Resultados/roc_curve.png)\n")
    print("Reporte generado en Resultados/reporte.md")