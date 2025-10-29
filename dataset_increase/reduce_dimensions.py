from neo4j import GraphDatabase
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def reduce_dimensions():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Crear carpeta reducers si no existe
    os.makedirs("reducers", exist_ok=True)
    print("Carpeta 'reducers/' creada o ya existe")
    
    try:
        with driver.session() as session:
            # Obtener noticias de entrenamiento con cluster_id
            print("Obteniendo noticias de entrenamiento...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train' AND n.cluster_id IS NOT NULL
                RETURN n.cluster_id AS cluster_id, n.embedding AS embedding, n.popularity AS popularity
            """)
            
            # Agrupar por cluster_id
            clusters_data = {}
            
            for record in result:
                cluster_id = record["cluster_id"]
                embedding = record["embedding"]
                popularity = record["popularity"]
                
                if cluster_id not in clusters_data:
                    clusters_data[cluster_id] = {"embeddings": [], "popularities": []}
                
                clusters_data[cluster_id]["embeddings"].append(embedding)
                clusters_data[cluster_id]["popularities"].append(popularity)
            
            print(f"Obtenidos datos de {len(clusters_data)} clústeres")
            
            processed_clusters = 0
            skipped_clusters = 0
            
            # Procesar cada clúster
            for cluster_id in sorted(clusters_data.keys()):
                try:
                    embeddings = np.array(clusters_data[cluster_id]["embeddings"])
                    popularities = np.array(clusters_data[cluster_id]["popularities"], dtype=int)
                    
                    n_samples = len(embeddings)
                    
                    # Verificar tamaño mínimo del clúster
                    if n_samples < 20:
                        print(f"⚠️  Clúster {cluster_id}: Solo {n_samples} noticias (mínimo 20). Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Verificar que hay variabilidad en las etiquetas
                    unique_labels = np.unique(popularities)
                    if len(unique_labels) < 2:
                        print(f"⚠️  Clúster {cluster_id}: Solo una clase de popularidad. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    print(f"\n🔄 Procesando Clúster {cluster_id}:")
                    print(f"   - Noticias: {n_samples}")
                    print(f"   - Dimensiones originales: {embeddings.shape[1]}")
                    print(f"   - Populares: {np.sum(popularities)} | No populares: {np.sum(~popularities)}")
                    
                    # Aplicar SelectKBest
                    k_features = min(20, embeddings.shape[1] // 2)  # Máximo 20 o la mitad de dimensiones
                    selector = SelectKBest(f_classif, k=k_features)
                    selector.fit(embeddings, popularities)
                    
                    # Obtener índices seleccionados
                    selected_indices = selector.get_support(indices=True)
                    
                    # Guardar el selector
                    selector_path = f"reducers/selector_cluster_{cluster_id}.pkl"
                    joblib.dump(selector, selector_path)
                    
                    print(f"   ✅ Características seleccionadas: {k_features}")
                    print(f"   📊 Muestra de índices: {selected_indices[:10]}...")
                    print(f"   💾 Guardado: {selector_path}")
                    
                    processed_clusters += 1
                    
                except Exception as e:
                    print(f"❌ Error procesando clúster {cluster_id}: {e}")
                    skipped_clusters += 1
                    continue
            
            # Resumen final
            print(f"\n🎯 Resumen del procesamiento:")
            print(f"   Total de clústeres encontrados: {len(clusters_data)}")
            print(f"   Clústeres procesados exitosamente: {processed_clusters}")
            print(f"   Clústeres saltados: {skipped_clusters}")
            
            # Verificar archivos guardados
            saved_files = [f for f in os.listdir("reducers") if f.startswith("selector_cluster_")]
            print(f"   Archivos guardados en 'reducers/': {len(saved_files)}")
            
            if saved_files:
                print("   📁 Archivos creados:")
                for file in sorted(saved_files):
                    print(f"      - {file}")
            
            print("\n✅ Proceso de reducción de dimensiones completado!")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\nConexión cerrada.")

if __name__ == "__main__":
    reduce_dimensions()