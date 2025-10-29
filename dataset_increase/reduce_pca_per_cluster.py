from neo4j import GraphDatabase
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os
import pandas as pd

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def reduce_pca_per_cluster():
    # Conectar a Neo4j
    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Crear carpeta pca_models si no existe
    os.makedirs("pca_models", exist_ok=True)
    print("📁 Carpeta 'pca_models/' creada o ya existe")
    
    # Valores de componentes a probar
    n_components_list = [20, 50, 100, 200, 500]
    print(f"🔍 Componentes PCA a evaluar: {n_components_list}")
    
    try:
        with driver.session() as session:
            # Obtener noticias de entrenamiento con cluster_id
            print("\n📊 Obteniendo noticias de entrenamiento...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train' AND n.cluster_id IS NOT NULL
                RETURN n.cluster_id AS cluster_id, n.embedding AS embedding
            """)
            
            # Agrupar por cluster_id
            clusters_data = {}
            total_news = 0
            
            for record in result:
                cluster_id = record["cluster_id"]
                embedding = record["embedding"]
                
                if cluster_id not in clusters_data:
                    clusters_data[cluster_id] = []
                
                clusters_data[cluster_id].append(embedding)
                total_news += 1
            
            print(f"✅ Obtenidos datos de {len(clusters_data)} clústeres con {total_news} noticias total")
            
            # Lista para almacenar resultados de varianza explicada
            variance_results = []
            cluster_best_pca = {}
            
            # Procesar cada clúster
            for cluster_id in sorted(clusters_data.keys()):
                embeddings = np.array(clusters_data[cluster_id])
                n_samples, n_features = embeddings.shape
                
                print(f"\n🔄 Procesando Clúster {cluster_id}:")
                print(f"   📈 Noticias: {n_samples}")
                print(f"   📐 Dimensiones originales: {n_features}")
                
                cluster_results = []
                
                # Probar diferentes valores de n_components
                for n_comp in n_components_list:
                    try:
                        # Ajustar n_components si es mayor que las muestras o características disponibles
                        max_components = min(n_samples - 1, n_features)
                        actual_n_comp = min(n_comp, max_components)
                        
                        if actual_n_comp <= 0:
                            print(f"   ⚠️  Componentes {n_comp}: No hay suficientes datos. Saltando...")
                            continue
                        
                        # Aplicar PCA
                        pca = PCA(n_components=actual_n_comp, random_state=42)
                        pca.fit(embeddings)
                        
                        # Calcular varianza explicada acumulada
                        explained_variance = np.sum(pca.explained_variance_ratio_)
                        
                        # Guardar modelo PCA
                        model_filename = f"pca_models/cluster_{cluster_id}_{actual_n_comp}.pkl"
                        joblib.dump(pca, model_filename)
                        
                        print(f"   ✅ PCA {actual_n_comp} componentes: {explained_variance:.4f} varianza explicada")
                        if actual_n_comp != n_comp:
                            print(f"      (ajustado de {n_comp} a {actual_n_comp} componentes)")
                        
                        # Guardar resultados
                        variance_results.append({
                            'cluster_id': cluster_id,
                            'n_components': actual_n_comp,
                            'explained_variance_ratio': explained_variance,
                            'n_samples': n_samples,
                            'original_features': n_features
                        })
                        
                        cluster_results.append((actual_n_comp, explained_variance))
                        
                    except Exception as e:
                        print(f"   ❌ Error con {n_comp} componentes: {e}")
                        continue
                
                # Encontrar el mejor PCA para este clúster (mayor varianza explicada)
                if cluster_results:
                    best_comp, best_variance = max(cluster_results, key=lambda x: x[1])
                    cluster_best_pca[cluster_id] = (best_comp, best_variance)
                    print(f"   🏆 Mejor configuración: {best_comp} componentes ({best_variance:.4f} varianza)")
            
            # Crear DataFrame con resultados
            if variance_results:
                df_results = pd.DataFrame(variance_results)
                
                # Guardar CSV
                csv_filename = 'pca_explained_variance.csv'
                df_results.to_csv(csv_filename, index=False)
                print(f"\n💾 Resultados guardados en '{csv_filename}'")
                
                # Mostrar tabla resumen
                print(f"\n📊 Resumen de varianza explicada por clúster:")
                print("="*80)
                print(f"{'Clúster':<8} {'Componentes':<12} {'Varianza':<10} {'Muestras':<9} {'Features':<9}")
                print("="*80)
                
                for _, row in df_results.iterrows():
                    print(f"{row['cluster_id']:<8} {row['n_components']:<12} "
                          f"{row['explained_variance_ratio']:<10.4f} {row['n_samples']:<9} "
                          f"{row['original_features']:<9}")
                
                print("="*80)
                
                # Mostrar mejores configuraciones por clúster
                print(f"\n🏆 Mejores configuraciones PCA por clúster:")
                print("-" * 60)
                for cluster_id in sorted(cluster_best_pca.keys()):
                    best_comp, best_variance = cluster_best_pca[cluster_id]
                    percentage = best_variance * 100
                    print(f"Clúster {cluster_id} → mejor PCA: {best_comp} componentes ({percentage:.1f}% varianza explicada)")
                
                # Estadísticas globales
                print(f"\n📈 Estadísticas globales:")
                avg_variance = df_results['explained_variance_ratio'].mean()
                max_variance = df_results['explained_variance_ratio'].max()
                min_variance = df_results['explained_variance_ratio'].min()
                
                print(f"   📊 Varianza explicada promedio: {avg_variance:.4f} ({avg_variance*100:.1f}%)")
                print(f"   📈 Varianza explicada máxima: {max_variance:.4f} ({max_variance*100:.1f}%)")
                print(f"   📉 Varianza explicada mínima: {min_variance:.4f} ({min_variance*100:.1f}%)")
                
                # Contar modelos guardados
                saved_models = [f for f in os.listdir("pca_models") if f.startswith("cluster_") and f.endswith(".pkl")]
                print(f"   💾 Total de modelos PCA guardados: {len(saved_models)}")
                
                # Mostrar algunos archivos guardados
                print(f"\n📁 Archivos PCA guardados (muestra):")
                for model_file in sorted(saved_models)[:10]:  # Mostrar primeros 10
                    print(f"   - {model_file}")
                if len(saved_models) > 10:
                    print(f"   ... y {len(saved_models) - 10} más")
                
            else:
                print("❌ No se pudieron generar resultados de PCA")
            
            print(f"\n✅ Proceso completado:")
            print(f"   🔢 Clústeres procesados: {len(clusters_data)}")
            print(f"   📊 Componentes evaluados: {n_components_list}")
            print(f"   📁 Carpeta de modelos: pca_models/")
            print(f"   📋 Archivo de resultados: pca_explained_variance.csv")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión cerrada.")

if __name__ == "__main__":
    reduce_pca_per_cluster()