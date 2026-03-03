import os
import skops.io as sio
from gensim.models import Word2Vec

os.makedirs("Modelo", exist_ok=True)


def guardar_modelo(modelo, nombre_modelo: str):
    """Guarda el clasificador con skops."""
    ruta = "Modelo/clasificador.skops"
    sio.dump(modelo, ruta)
    print(f"Modelo guardado: {ruta} ({nombre_modelo})")


def guardar_tfidf(tfidf):
    """Guarda el vectorizador TF-IDF con skops."""
    ruta = "Modelo/tfidf.skops"
    sio.dump(tfidf, ruta)
    print(f"TF-IDF guardado: {ruta}")


def guardar_word2vec(modelo_w2v: Word2Vec):
    """Guarda el modelo Word2Vec en formato nativo de Gensim."""
    ruta = "Modelo/word2vec.model"
    modelo_w2v.save(ruta)
    print(f"Word2Vec guardado: {ruta}")


def cargar_modelo(nombre_modelo: str):
    """Carga el clasificador desde skops."""
    ruta = "Modelo/clasificador.skops"
    return sio.load(ruta, trusted=sio.get_untrusted_types(file=ruta))


def cargar_tfidf():
    """Carga el vectorizador TF-IDF."""
    ruta = "Modelo/tfidf.skops"
    return sio.load(ruta, trusted=sio.get_untrusted_types(file=ruta))


def cargar_word2vec():
    """Carga el modelo Word2Vec."""
    ruta = "Modelo/word2vec.model"
    return Word2Vec.load(ruta)