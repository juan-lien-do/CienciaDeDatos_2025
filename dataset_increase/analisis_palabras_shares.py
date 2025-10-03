import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import re

# Palabras a analizar (en minúsculas)
palabras = [
    'google', 'facebook', 'twitter', 'apple', 'world', 'video', 'app', 'watch', 'iphone', 'game'
]

# Leer los CSV
shares_csv = './OnlineNewsPopularity.csv'  # Debe tener columnas 'url' y 'shares'
titulos_csv = './AugmentedNews.csv'        # Debe tener columnas 'url' y 'titulo'
df_shares = pd.read_csv(shares_csv)
df_titulos = pd.read_csv(titulos_csv)


# Normalizar URLs antes del merge
def normalizar_url(url):
    url = str(url).strip().lower().replace('"', '')
    url = url.replace('https://', '').replace('http://', '')
    if url.endswith('/'):
        url = url[:-1]
    url = url.split('?')[0]  # Elimina parámetros
    return url

df_shares['url_norm'] = df_shares['url'].apply(normalizar_url)
df_titulos['url_norm'] = df_titulos['url'].apply(normalizar_url)

# Mostrar las primeras 10 URLs normalizadas de cada archivo para comparar
print('\nPrimeras 10 URLs normalizadas de AugmentedNews.csv:')
print(df_titulos['url_norm'].head(10).to_list())
print('\nPrimeras 10 URLs normalizadas de OnlineNewsPopularity.csv:')
print(df_shares['url_norm'].head(10).to_list())


# Normalizar URLs antes del merge
def normalizar_url(url):
    url = str(url).strip().lower().replace('"', '')
    url = url.replace('https://', '').replace('http://', '')
    if url.endswith('/'):
        url = url[:-1]
    url = url.split('?')[0]  # Elimina parámetros
    return url

df_shares['url_norm'] = df_shares['url'].apply(normalizar_url)
df_titulos['url_norm'] = df_titulos['url'].apply(normalizar_url)

# Unir ambos DataFrames por 'url_norm'
df = pd.merge(df_titulos, df_shares, on='url_norm', how='inner')
print(f'Filas después del merge: {len(df)}')
print(f'Columnas disponibles: {df.columns.tolist()}')


# Detectar el nombre correcto de la columna de compartidos
col_shares = None
for col in df.columns:
    if col.strip().lower() == 'shares':
        col_shares = col
        break
if col_shares is None:
    raise ValueError('No se encontró la columna de compartidos (shares) en el DataFrame.')

# Resultados
resultados = []
for palabra in palabras:
    pattern = re.compile(rf'\b{re.escape(palabra)}\b', re.IGNORECASE)
    indices = df['titulo'].dropna().apply(lambda t: bool(pattern.search(str(t))))
    subset = df[indices]
    cantidad = subset.shape[0]
    promedio = subset[col_shares].mean() if cantidad > 0 else 0
    resultados.append({'palabra': palabra.capitalize(), 'promedio_compartidos': promedio, 'cantidad_apariciones': cantidad})

# Crear DataFrame de resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)


# Visualización: gráfico de línea con puntos
plt.figure(figsize=(12, 6))
x = range(len(df_resultados))

plt.plot(x, df_resultados['promedio_compartidos'], marker='o', label='Promedio Compartidos', color='skyblue')
plt.plot(x, df_resultados['cantidad_apariciones'], marker='o', label='Cantidad Apariciones', color='orange')

plt.xticks(x, df_resultados['palabra'], rotation=45)
plt.xlabel('Palabra', fontsize=16)
plt.ylabel('Valor', fontsize=16)
plt.title('Promedio de compartidos y cantidad de apariciones por palabra en títulos', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
