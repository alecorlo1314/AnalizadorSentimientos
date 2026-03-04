# 🎬 Análisis de Sentimientos en Reseñas de Películas en Español

Proyecto de NLP para clasificar reseñas de películas en español como **positivas o negativas** usando el dataset IMDB 50K traducido al español. Incluye pipeline completo con CI/CD, versionado de datos con DVC y explicabilidad con LIME.

🚀 **[Demo en vivo en Hugging Face Spaces](https://huggingface.co/spaces/alecorlo1234/AnalizadorSentimientos)**

---

## 📊 Resultados del Modelo

| Métrica | Valor |
|---|---|
| Algoritmo | TF-IDF + Logistic Regression |
| F1-Score (CV) | 0.8779 |
| F1-Score (Test) | 0.8793 |
| Accuracy (Test) | 0.8765 |
| Dataset | 49,599 reseñas · balanceado |

### Comparación de algoritmos evaluados

| Algoritmo | F1-Score (CV) |
|---|---|
| **TF-IDF + LogisticRegression** ✅ | **0.8779** |
| TF-IDF + LinearSVC | evaluado |
| Word2Vec + LogisticRegression | evaluado |
| Word2Vec + LinearSVC | evaluado |

La selección es automática — el pipeline compara los 4 y elige el mejor por F1-Score via cross-validation de 5 folds.

---

## 🏗️ Arquitectura del Proyecto

```
AnalizadorSentimientos/
├── src/
│   ├── datos.py        # Carga y división del dataset
│   ├── preprocesar.py  # Limpieza, stemming con NLTK y cache
│   ├── entrenar.py     # Comparación TF-IDF vs Word2Vec + MLflow tracking
│   ├── evaluar.py      # Métricas, matriz de confusión, curva ROC
│   ├── explicar.py     # Explicabilidad LIME por predicción
│   └── guardar.py      # Serialización con skops
├── Aplicacion/
│   ├── app.py          # App Gradio con análisis y LIME visual
│   ├── requirements.txt
│   └── README.md       # Configuración para Hugging Face Spaces
├── Modelo/
│   ├── clasificador.skops  # Modelo serializado
│   └── tfidf.skops         # Vectorizador TF-IDF
├── Datos/              # Gestionado por DVC (no en Git)
├── Resultados/         # Generado en CI (no en Git)
├── .github/
│   ├── workflows/ci.yml   # Entrenamiento + reporte automático
│   └── workflows/cd.yml   # Deploy a Hugging Face
├── entrenamiento.py    # Script principal del pipeline
├── Makefile
└── requirements.txt
```

---

## ⚙️ Pipeline de CI/CD

```
Push a main
    │
    ▼
┌─────────────────────────────────────┐
│         Continuous Integration       │
│  format → lint → DVC pull → train   │
│  → eval → reporte en PR → push      │
│         al branch update            │
└──────────────────┬──────────────────┘
                   │ éxito
                   ▼
┌─────────────────────────────────────┐
│        Continuous Deployment         │
│  checkout update → login HF →       │
│  upload Aplicacion/ + Modelo/        │
│       + src/ a Hugging Face Spaces   │
└─────────────────────────────────────┘
```

---

## 🚀 Instalación y uso local

### 1. Clonar el repositorio

```bash
git clone https://github.com/alecorlo1314/AnalizadorSentimientos.git
cd AnalizadorSentimientos
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Descargar recursos NLTK

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 4. Configurar DVC y descargar datos

Necesitas una cuenta en [DagsHub](https://dagshub.com).

```bash
# Linux/Mac
export DAGSHUB_TOKEN=tu_token_aqui

# Windows (PowerShell)
$env:DAGSHUB_TOKEN="tu_token_aqui"

dvc remote modify sentimiento_storage password $DAGSHUB_TOKEN
dvc pull -r sentimiento_storage
```

### 5. Entrenar el modelo

```bash
python entrenamiento.py
```

Esto compara 4 combinaciones (TF-IDF vs Word2Vec) × (LogisticRegression vs LinearSVC) via cross-validation y guarda el mejor en `Modelo/`.

### 6. Ver experimentos en MLflow

```bash
mlflow ui
```

Abre `http://localhost:5000` para comparar todos los runs con métricas y parámetros.

### 7. Correr la app localmente

```bash
cd Aplicacion
pip install -r requirements.txt
python app.py
```

Abre `http://localhost:7860` en el navegador.

---

## 🔐 Secrets de GitHub necesarios

Configura estos secrets en **Settings → Secrets and variables → Actions**:

| Secret | Descripción |
|---|---|
| `DAGSHUB_TOKEN` | Token de API de DagsHub |
| `HF_SENTIMIENTOS` | Token de Hugging Face con permisos de escritura |
| `USER_NAME` | Tu nombre para los commits automáticos |
| `USER_EMAIL` | Tu email para los commits automáticos |

> `GITHUB_TOKEN` se genera automáticamente.

---

## 📋 Comandos disponibles (Makefile)

```bash
make install              # Instalar dependencias
make format               # Verificar formato con black
make lint                 # Analizar calidad con pylint
make train                # Entrenar modelo
make eval                 # Evaluar y generar reporte
make configuracion_DVC_remoto  # Configurar remote de DagsHub
make deploy HF=<token>    # Deploy manual a Hugging Face
```

---

## 🔍 Técnicas de NLP aplicadas

**Preprocesamiento** — limpieza de HTML y caracteres especiales, tokenización, eliminación de stopwords en español y stemming con NLTK SnowballStemmer. Cache automático para no reprocesar en cada ejecución.

**TF-IDF** — representa cada reseña como un vector de frecuencias de términos ponderadas por su rareza en el corpus. Captura qué palabras son importantes en cada documento.

**Word2Vec** — entrena embeddings desde cero sobre el corpus, aprendiendo representaciones semánticas donde palabras similares quedan cerca en el espacio vectorial.

**MLflow** — tracking automático de todos los experimentos con parámetros, métricas y duración. UI web para comparar runs visualmente.

**LIME** — explicabilidad local por predicción. Para cada reseña muestra las palabras que más influyeron en la decisión del modelo, con dirección (positivo/negativo).

---

## 📦 Tecnologías utilizadas

- **NLP**: NLTK, Gensim (Word2Vec)
- **ML**: scikit-learn
- **Tracking**: MLflow
- **Explicabilidad**: LIME
- **Versionado de datos**: DVC + DagsHub
- **Serialización**: skops
- **App**: Gradio
- **CI/CD**: GitHub Actions + CML
- **Deploy**: Hugging Face Spaces

---

## ⚠️ Nota

Este proyecto es de carácter **educativo**. El dataset usa traducciones automáticas al español lo que puede afectar la calidad del preprocesamiento en textos nativos en español.
