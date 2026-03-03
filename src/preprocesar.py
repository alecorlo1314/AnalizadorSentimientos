import re
import os
import pickle
import pandas as pd
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

# Descargar recursos NLTK si no existen
nltk.download("stopwords", quiet=True)

stemmer = SnowballStemmer("spanish")
STOPWORDS = set(stopwords.words("spanish")) | {
    "si",
    "así",
    "tan",
    "también",
    "pero",
    "porque",
    "cuando",
    "donde",
    "como",
    "aunque",
    "sino",
    "ya",
    "aún",
    "bien",
    "mal",
    "solo",
    "ser",
    "estar",
    "haber",
    "tener",
    "hacer",
    "decir",
    "ver",
    "dar",
    "saber",
    "querer",
    "llegar",
    "película",
    "film",
}

CACHE_PATH = "Datos/tokens_cache.pkl"


def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto.lower())
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list[str]:
    return [
        stemmer.stem(palabra)
        for palabra in texto.split()
        if palabra not in STOPWORDS and len(palabra) > 2
    ]


def preprocesar_serie(serie: pd.Series, verbose=True, usar_cache=True) -> pd.Series:
    """
    Preprocesa una Serie de textos con stemming NLTK.
    Cachea el resultado en disco para no reprocesar en cada ejecución.
    """
    # Intentar cargar desde cache
    cache_key = f"{CACHE_PATH}.{len(serie)}.{serie.index[0]}"
    cache_file = f"{CACHE_PATH}_{len(serie)}_{hash(str(serie.index[0]))}.pkl"

    if usar_cache and os.path.exists(cache_file):
        print(f"Cargando tokens desde cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Procesar
    iterador = serie
    if verbose:
        tqdm.pandas(desc="Preprocesando")
        resultados = serie.progress_apply(lambda t: tokenizar(limpiar_texto(t)))
    else:
        resultados = serie.apply(lambda t: tokenizar(limpiar_texto(t)))

    # Guardar cache
    if usar_cache:
        os.makedirs("Datos", exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(resultados, f)
        print(f"Cache guardado: {cache_file}")

    return resultados


def tokens_a_texto(tokens: list[str]) -> str:
    return " ".join(tokens)
