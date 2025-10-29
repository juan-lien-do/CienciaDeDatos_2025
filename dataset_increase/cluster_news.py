from neo4j import GraphDatabase
import numpy as np
from sklearn.cluster import KMeans
import pickle

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def cluster_news():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Obtener noticias de entrenamiento
            print("Obteniendo noticias de entrenamiento...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train'
                RETURN n.url AS url, n.embedding AS embedding
            """)
            
            # Recolectar datos
            urls = []
            embeddings = []
            
            for record in result:
                urls.append(record["url"])
                embeddings.append(record["embedding"])
            
            if not embeddings:
                print("No se encontraron noticias de entrenamiento.")
                return
            
            print(f"Obtenidas {len(embeddings)} noticias de entrenamiento")
            
            # Convertir embeddings a NumPy array
            print("Convirtiendo embeddings a array NumPy...")
            X = np.array(embeddings)
            print(f"Shape del array: {X.shape}")
            
            # Ejecutar K-Means
            print("Ejecutando K-Means clustering...")
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            print("Clustering completado!")
            
            # Preparar datos para actualización por lotes
            print("Actualizando cluster_id en Neo4j...")
            batch_size = 500
            total = len(urls)
            
            for i in range(0, total, batch_size):
                batch_urls = urls[i:i+batch_size]
                batch_labels = cluster_labels[i:i+batch_size]
                
                # Actualizar por lotes
                with driver.session() as batch_session:
                    for url, cluster_id in zip(batch_urls, batch_labels):
                        batch_session.run("""
                            MATCH (n:Noticia {url: $url})
                            SET n.cluster_id = $cluster_id
                        """, url=url, cluster_id=int(cluster_id))
                
                processed = min(i + batch_size, total)
                print(f"Procesadas {processed}/{total} noticias")
            
            # Contar noticias por clúster
            print("\n📊 Distribución de noticias por clúster:")
            for cluster_id in range(10):
                count_result = session.run("""
                    MATCH (n:Noticia)
                    WHERE n.subset = 'train' AND n.cluster_id = $cluster_id
                    RETURN count(n) AS count
                """, cluster_id=cluster_id)
                
                count = count_result.single()["count"]
                percentage = (count / total) * 100
                print(f"  Clúster {cluster_id}: {count} noticias ({percentage:.1f}%)")
            
            # Guardar el modelo entrenado
            print("\nGuardando modelo K-Means...")
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump(kmeans, f)
            print("Modelo guardado como 'kmeans_model.pkl'")
            
            print(f"\n✅ Proceso completado:")
            print(f"   Total de noticias procesadas: {total}")
            print(f"   Número de clústeres: 10")
            print(f"   Modelo guardado: kmeans_model.pkl")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\nConexión cerrada.")

if __name__ == "__main__":
    cluster_news()