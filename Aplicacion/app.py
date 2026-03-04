import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import skops.io as sio
from src.preprocesar import limpiar_texto, tokenizar, tokens_a_texto
from src.explicar import explicar_prediccion, graficar_explicacion

# ── Cargar modelo y vectorizador ───────────────────────────────────────────────
MODELO_PATH = "Modelo/clasificador.skops"
TFIDF_PATH  = "Modelo/tfidf.skops"

unsafe_clasi = sio.get_untrusted_types(file=MODELO_PATH)
unsafe_tfidf = sio.get_untrusted_types(file=TFIDF_PATH)
modelo = sio.load(MODELO_PATH, trusted=unsafe_clasi)
tfidf = sio.load(TFIDF_PATH, trusted=unsafe_tfidf)


# ── Función principal ──────────────────────────────────────────────────────────
def analizar(texto: str):
    if not texto or not texto.strip():
        return (
            '<div style="color:#888;text-align:center;padding:20px;">Escribe una reseña para analizar.</div>',
            None,
        )

    # Preprocesar
    tokens = tokenizar(limpiar_texto(texto))
    texto_limpio = tokens_a_texto(tokens)

    if not texto_limpio.strip():
        return (
            '<div style="color:#e0a020;text-align:center;padding:20px;">⚠ El texto quedó vacío tras el preprocesamiento. Intenta con más palabras.</div>',
            None,
        )

    # Predicción
    X = tfidf.transform([texto_limpio])
    proba = modelo.predict_proba(X)[0]
    proba_pos = proba[1]
    proba_neg = proba[0]
    es_positivo = proba_pos >= 0.5

    etiqueta = "😊 POSITIVO" if es_positivo else "😞 NEGATIVO"
    color = "#4caf7d" if es_positivo else "#e05c5c"
    descripcion = (
        "La reseña expresa una opinión favorable."
        if es_positivo
        else "La reseña expresa una opinión desfavorable."
    )

    # Barra de probabilidad
    barra_pos = f'<div style="height:8px;width:{proba_pos*100:.0f}%;background:#4caf7d;border-radius:4px 0 0 4px;display:inline-block;"></div>'
    barra_neg = f'<div style="height:8px;width:{proba_neg*100:.0f}%;background:#e05c5c;border-radius:0 4px 4px 0;display:inline-block;"></div>'

    resultado_html = f"""
    <div style="background:#1a1d27;border-radius:12px;padding:24px;border:1px solid #2a2d3a;">
        <div style="text-align:center;margin-bottom:16px;">
            <div style="font-size:2rem;font-weight:700;color:{color};">{etiqueta}</div>
            <div style="color:#aaa;font-size:0.9rem;margin-top:4px;">{descripcion}</div>
        </div>
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;color:#ccc;font-size:0.85rem;margin-bottom:4px;">
                <span>😊 Positivo: <b style="color:#4caf7d;">{proba_pos:.1%}</b></span>
                <span>😞 Negativo: <b style="color:#e05c5c;">{proba_neg:.1%}</b></span>
            </div>
            <div style="background:#0f1117;border-radius:4px;overflow:hidden;height:8px;display:flex;">
                {barra_pos}{barra_neg}
            </div>
        </div>
        <div style="color:#555;font-size:0.75rem;text-align:center;margin-top:8px;">
            Tokens procesados: {len(tokens)} palabras
        </div>
    </div>
    """

    # LIME
    try:
        _, palabras_pesos = explicar_prediccion(
            texto_limpio, tfidf, modelo, num_features=10, num_samples=300
        )
        fig = graficar_explicacion(palabras_pesos, titulo="Palabras que más influyeron")
    except Exception as e:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.text(
            0.5,
            0.5,
            f"LIME no disponible:\n{e}",
            ha="center",
            va="center",
            color="white",
            transform=ax.transAxes,
        )
        ax.axis("off")

    return resultado_html, fig


# ── UI ─────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

body, .gradio-container {
    background: #0b0d14 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e0e0e0 !important;
}
.gr-panel, .gr-box, .gr-form {
    background: #13151f !important;
    border: 1px solid #1f2235 !important;
    border-radius: 10px !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #4caf7d, #2d8f5e) !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    color: white !important;
}
textarea {
    background: #0f1117 !important;
    border: 1px solid #2a2d3a !important;
    color: #e0e0e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
footer { display: none !important; }
"""

EJEMPLOS = [
    [
        "Esta película es una obra maestra del cine moderno. Los actores están increíbles y la historia me dejó sin palabras. La recomiendo completamente."
    ],
    [
        "Qué pérdida de tiempo tan terrible. La trama no tiene sentido, los actores actúan fatal y el final fue decepcionante. No la vean."
    ],
    [
        "El film tiene sus momentos buenos y malos. La fotografía es bonita pero la historia es confusa. En general es una película del montón."
    ],
]

with gr.Blocks(css=CSS, title="Análisis de Sentimientos") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 16px;border-bottom:1px solid #1f2235;margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:0.75rem;letter-spacing:0.2em;color:#4caf7d;margin-bottom:8px;">
            NLP · ANÁLISIS DE SENTIMIENTOS
        </div>
        <h1 style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;color:#fff;margin:0;">
            ¿Qué siente tu reseña?
        </h1>
        <p style="color:#888;margin-top:10px;font-size:0.9rem;max-width:520px;margin-inline:auto;">
            Escribe una reseña de película en español y el modelo detectará si es positiva o negativa,
            mostrando qué palabras influyeron en la decisión.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            texto_input = gr.Textbox(
                label="✍️ Escribe tu reseña",
                placeholder="Ej: Esta película me pareció increíble, la historia es fascinante...",
                lines=6,
                max_lines=12,
            )
            btn = gr.Button("🔍 Analizar sentimiento", variant="primary", size="lg")
            gr.Examples(
                examples=EJEMPLOS,
                inputs=texto_input,
                label="💡 Ejemplos",
            )

        with gr.Column(scale=1):
            resultado = gr.HTML(
                value="""<div style="background:#1a1d27;border-radius:12px;padding:30px;
                text-align:center;border:1px solid #2a2d3a;color:#555;">
                El resultado aparecerá aquí</div>"""
            )
            gr.HTML(
                "<div style='margin-top:12px;font-family:Syne,sans-serif;font-size:0.7rem;color:#444;letter-spacing:0.12em;'>EXPLICABILIDAD · LIME</div>"
            )
            lime_plot = gr.Plot()

    btn.click(fn=analizar, inputs=texto_input, outputs=[resultado, lime_plot])
    texto_input.submit(fn=analizar, inputs=texto_input, outputs=[resultado, lime_plot])

    gr.HTML("""
    <div style="text-align:center;padding:20px 0;margin-top:20px;border-top:1px solid #1f2235;">
        <span style="font-family:'Syne',sans-serif;font-size:0.7rem;color:#333;letter-spacing:0.15em;">
            MODELO: TF-IDF + LOGISTIC REGRESSION · DATASET: IMDB 50K EN ESPAÑOL · EXPLICABILIDAD: LIME
        </span>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
