import re
import pandas as pd
from tqdm import tqdm # Para mostrar barras de progreso
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

def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto) # Eliminar etiquetas HTML
    texto = re.sub(r"http\S+|www\S+", " ", texto) # Eliminar URLs
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto.lower()) # Eliminar caracteres especiales y los convertir a minúsculas
    texto = re.sub(r"\s+", " ", texto).strip() # Eliminar espacios en blanco
    return texto


def tokenizar(texto: str) -> list[str]:
    return [
        stemmer.stem(palabra) #Reduce la palabra a su raíz ejemplo "corriendo", "corrí", "corre" -> "corr"
        for palabra in texto.split()# Separa el texto en palabras individuales cuando hay espacios
        if palabra not in STOPWORDS and len(palabra) > 2 # Eliminar stopwords y palabras muy cortas
    ]


def preprocesar_serie(serie: pd.Series, verbose=True, usar_cache=True) -> pd.Series:
    """
    Preprocesa una Serie de textos con stemming NLTK.
    Cachea el resultado en disco para no reprocesar en cada ejecución.
    """
    # Procesar
    if verbose:
        tqdm.pandas(desc="Preprocesando") #Mosntrar barra de progreso al usar apply o progress_apply
        resultados = serie.progress_apply(lambda t: tokenizar(limpiar_texto(t)))
    else:
        resultados = serie.apply(lambda t: tokenizar(limpiar_texto(t)))

    return resultados


def tokens_a_texto(tokens: list[str]) -> str:
    return " ".join(tokens)
