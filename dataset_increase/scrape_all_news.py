import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

# Leer el CSV original
input_csv = './OnlineNewsPopularity.csv'
df = pd.read_csv(input_csv)

# Scrappear todas las URLs
def scrape_news(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extraer el título
        if soup.title:
            titulo = soup.title.get_text(strip=True)
        else:
            h1 = soup.find('h1')
            titulo = h1.get_text(strip=True) if h1 else 'Sin título'
        # Extraer los párrafos principales
        parrafos = soup.find_all('p')
        texto = ' '.join([
            p.get_text(strip=True).replace('\n', ' ').replace('\r', ' ')
            for p in parrafos if p.get_text(strip=True)
        ])
    except Exception as e:
        titulo = 'ERROR'
        texto = f'Error al scrapear: {e}'
    return {'url': url, 'titulo': titulo, 'texto': texto}


output_csv = './AugmentedNews.csv'
resultados = []
batch_size = 100
total = len(df)

for idx, row in df.iterrows():
    url = row['url']
    resultados.append(scrape_news(url))
    print(f"Procesada {idx+1}/{total}: {url}")
    # Guardar cada batch_size noticias
    if (idx + 1) % batch_size == 0 or (idx + 1) == total:
        mode = 'a' if idx + 1 > batch_size else 'w'
        header = (idx + 1 <= batch_size)
        pd.DataFrame(resultados).to_csv(
            output_csv,
            mode=mode,
            header=header,
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            lineterminator='\n'
        )
        print(f'Guardadas {idx+1} noticias en {output_csv}')
        resultados = []  # Liberar memoria

print(f'Scraping completado para {total} noticias. Resultado guardado en {output_csv}')
