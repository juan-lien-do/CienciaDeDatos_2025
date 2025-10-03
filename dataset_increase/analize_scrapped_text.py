import pandas as pd
from collections import Counter
import re

# Leer el CSV generado por el scraping
df = pd.read_csv('./AugmentedNews.csv')

# Contar filas con campos vacíos en 'titulo' o 'texto'
campos_vacios = df['titulo'].isna().sum() + df['texto'].isna().sum()
print(f'Filas con campos vacíos en "titulo" o "texto": {campos_vacios}')

# Tokenizar títulos y contar palabras
titulos = df['titulo'].dropna().astype(str)
palabras = []
for titulo in titulos:
    # Eliminar signos de puntuación y pasar a minúsculas
    tokens = re.findall(r'\b\w+\b', titulo.lower())
    palabras.extend(tokens)

contador = Counter(palabras)
mas_comunes = contador.most_common(100)

print("Las 100 palabras más comunes en los títulos:")
for palabra, cantidad in mas_comunes:
    print(f"{palabra}: {cantidad}")