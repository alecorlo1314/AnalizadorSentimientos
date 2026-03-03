import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.

    Args:
        ruta (str): La ruta al archivo CSV.

    Returns:
        pd.DataFrame: Un DataFrame con los datos cargados.
    """
    # Cargarmanos el archivo CSV y quitamos la columna 0
    df = pd.read_csv(ruta, index_col=0)
    # Seleccionamos solo las columnas necesarias y renombramos
    df = df[["review_es", "sentimiento"]].copy()
    df.columns = ["texto", "etiqueta"]
    # Mapeamos la etiqueta con valores numericos (1 para positivo, 0 para negativo)
    df["etiqueta"] = df["etiqueta"].map({"positivo": 1, "negativo": 0})
    # Eliminamos filas con texto vacío o nulo
    df = df.dropna()
    df = df[df["texto"].str.strip() != ""]
    # Eliminamos filas duplicadas
    df = df.drop_duplicates()
    return df


def separar_features(df: pd.DataFrame):
    X = df["texto"]
    y = df["etiqueta"]
    return X, y


def dividir_datos(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
