import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

# Leer el CSV original
input_csv = './OnlineNewsPopularity.csv'
df = pd.read_csv(input_csv)

# Tomar la primera URL
first_url = df.iloc[0]['url']

# Hacer la petición HTTP
try:
    response = requests.get(first_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extraer el título
    titulo = ''
    if soup.title:
        titulo = soup.title.get_text(strip=True)
    else:
        h1 = soup.find('h1')
        titulo = h1.get_text(strip=True) if h1 else 'Sin título'
    # Extraer los párrafos principales
    parrafos = soup.find_all('p')
    texto = ' '.join([p.get_text(strip=True).replace('\n', ' ').replace('\r', ' ') for p in parrafos if p.get_text(strip=True)])
except Exception as e:
    titulo = 'ERROR'
    texto = f'Error al scrapear: {e}'

# Guardar en nuevo CSV
output_csv = './AugmentedNews.csv'
augmented_df = pd.DataFrame([{
    'url': first_url,
    'titulo': titulo,
    'texto': texto
}])
augmented_df.to_csv(
    output_csv,
    index=False,
    encoding='utf-8',
    quoting=csv.QUOTE_ALL,      # Fuerza comillas en todos los campos
    doublequote=True,           # Escapa comillas dobles internas correctamente
    lineterminator='\n'        # Fuerza saltos de línea tipo Unix
)

print(f'Scraping completado. Resultado guardado en {output_csv}')
