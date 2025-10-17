from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pandas as pd
import re

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password" 

# Cargar modelo de embeddings (modelo ligero y rápido)
print("Cargando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Conectar a Neo4j
print("Conectando a Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Leer los CSV
print("Leyendo archivos CSV...")
df_noticias = pd.read_csv('./AugmentedNews_Fixed.csv')
df_shares = pd.read_csv('./OnlineNewsPopularity.csv')

# Normalizar URLs para el merge
def normalizar_url(url):
    url = str(url).strip().lower().replace('"', '')
    url = url.replace('https://', '').replace('http://', '')
    if url.endswith('/'):
        url = url[:-1]
    url = url.split('?')[0]
    return url

df_noticias['url_norm'] = df_noticias['url'].apply(normalizar_url)
df_shares['url_norm'] = df_shares['url'].apply(normalizar_url)

# Hacer merge para obtener shares
print("Cruzando datos con shares...")
df = pd.merge(df_noticias, df_shares[['url_norm', ' shares']], on='url_norm', how='left')
df.rename(columns={' shares': 'shares'}, inplace=True)
df['shares'] = df['shares'].fillna(0).astype(int)

print(f"Total de noticias a procesar: {len(df)}")

# Crear constraint para evitar duplicados
with driver.session() as session:
    try:
        session.run("CREATE CONSTRAINT noticia_url IF NOT EXISTS FOR (n:Noticia) REQUIRE n.url IS UNIQUE")
        print("Constraint creado en Neo4j")
    except Exception as e:
        print(f"Constraint ya existe o error: {e}")

# Procesar y guardar en Neo4j por lotes
batch_size = 50
total = len(df)

for i in range(0, total, batch_size):
    batch = df.iloc[i:i+batch_size]
    
    with driver.session() as session:
        for idx, row in batch.iterrows():
            try:
                # Generar embedding del texto completo (titulo + texto)
                texto_completo = f"{row['titulo']} {row['texto']}"
                embedding = model.encode(texto_completo).tolist()
                
                # Guardar en Neo4j
                session.run("""
                    MERGE (n:Noticia {url: $url})
                    SET n.titulo = $titulo,
                        n.texto = $texto,
                        n.shares = $shares,
                        n.embedding = $embedding
                """, url=row['url'], 
                     titulo=row['titulo'], 
                     texto=row['texto'][:5000],  # Limitar texto para no sobrecargar
                     shares=int(row['shares']),
                     embedding=embedding)
                
            except Exception as e:
                print(f"Error procesando noticia {idx}: {e}")
    
    print(f"Procesadas {min(i+batch_size, total)}/{total} noticias")

driver.close()
print("¡Proceso completado! Todas las noticias han sido guardadas en Neo4j.")
