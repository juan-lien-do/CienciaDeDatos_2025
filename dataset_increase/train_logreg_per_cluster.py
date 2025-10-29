from neo4j import GraphDatabase
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import pandas as pd

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def train_logreg_per_cluster():
    # Conectar a Neo4j
    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Crear carpeta logreg_models si no existe
    os.makedirs("logreg_models", exist_ok=True)
    print("📁 Carpeta 'logreg_models/' creada o ya existe")
    
    # Verificar que existe el archivo de varianza explicada
    pca_variance_file = 'pca_explained_variance.csv'
    if not os.path.exists(pca_variance_file):
        print(f"❌ Error: No se encontró '{pca_variance_file}'. Ejecuta primero reduce_pca_per_cluster.py")
        return
    
    # Verificar que existe la carpeta de modelos PCA
    if not os.path.exists("pca_models"):
        print("❌ Error: La carpeta 'pca_models/' no existe. Ejecuta primero reduce_pca_per_cluster.py")
        return
    
    try:
        # Cargar datos de varianza explicada para encontrar los mejores PCA
        print("📊 Cargando datos de varianza explicada de PCA...")
        df_pca_variance = pd.read_csv(pca_variance_file)
        
        # Encontrar la mejor configuración PCA por clúster (mayor varianza explicada)
        best_pca_per_cluster = df_pca_variance.loc[df_pca_variance.groupby('cluster_id')['explained_variance_ratio'].idxmax()]
        best_pca_dict = dict(zip(best_pca_per_cluster['cluster_id'], best_pca_per_cluster['n_components']))
        
        print("🏆 Mejores configuraciones PCA por clúster:")
        for cluster_id, n_comp in best_pca_dict.items():
            variance = best_pca_per_cluster[best_pca_per_cluster['cluster_id'] == cluster_id]['explained_variance_ratio'].iloc[0]
            print(f"   Clúster {cluster_id}: {n_comp} componentes ({variance:.4f} varianza)")
        
        with driver.session() as session:
            # Obtener noticias de entrenamiento con cluster_id y popularity
            print("\n📊 Obteniendo noticias de entrenamiento...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train' AND n.cluster_id IS NOT NULL
                RETURN n.cluster_id AS cluster_id, n.embedding AS embedding, n.popularity AS popularity
            """)
            
            # Agrupar por cluster_id
            clusters_data = {}
            total_news = 0
            
            for record in result:
                cluster_id = record["cluster_id"]
                embedding = record["embedding"]
                popularity = record["popularity"]
                
                if cluster_id not in clusters_data:
                    clusters_data[cluster_id] = {"embeddings": [], "popularities": []}
                
                clusters_data[cluster_id]["embeddings"].append(embedding)
                clusters_data[cluster_id]["popularities"].append(popularity)
                total_news += 1
            
            print(f"✅ Obtenidos datos de {len(clusters_data)} clústeres con {total_news} noticias total")
            
            # Lista para almacenar métricas de entrenamiento
            training_metrics = []
            processed_clusters = 0
            skipped_clusters = 0
            
            # Procesar cada clúster
            for cluster_id in sorted(clusters_data.keys()):
                try:
                    embeddings = np.array(clusters_data[cluster_id]["embeddings"])
                    popularities = np.array(clusters_data[cluster_id]["popularities"], dtype=int)
                    n_samples = len(embeddings)
                    
                    print(f"\n🔄 Procesando Clúster {cluster_id}:")
                    print(f"   📈 Noticias: {n_samples}")
                    print(f"   📐 Dimensiones originales: {embeddings.shape[1]}")
                    
                    # Verificar tamaño mínimo del clúster
                    if n_samples < 10:
                        print(f"   ⚠️  Solo {n_samples} noticias (mínimo 10). Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Verificar que hay variabilidad en las etiquetas
                    unique_labels = np.unique(popularities)
                    if len(unique_labels) < 2:
                        print(f"   ⚠️  Solo una clase de popularidad. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Verificar que tenemos el mejor PCA para este clúster
                    if cluster_id not in best_pca_dict:
                        print(f"   ⚠️  No se encontró configuración PCA para este clúster. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Cargar el mejor modelo PCA para este clúster
                    best_n_comp = best_pca_dict[cluster_id]
                    pca_model_file = f"pca_models/cluster_{cluster_id}_{best_n_comp}.pkl"
                    
                    if not os.path.exists(pca_model_file):
                        print(f"   ⚠️  No se encontró modelo PCA: {pca_model_file}. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    pca_model = joblib.load(pca_model_file)
                    print(f"   ✅ PCA cargado: {best_n_comp} componentes")
                    
                    # Transformar embeddings con PCA
                    X_pca = pca_model.transform(embeddings)
                    print(f"   📊 Dimensiones después de PCA: {X_pca.shape[1]}")
                    
                    # Entrenar Regresión Logística
                    print("   🧠 Entrenando Regresión Logística...")
                    logreg = LogisticRegression(
                        solver='liblinear',
                        C=1.0,
                        penalty='l2',
                        max_iter=1000,
                        random_state=42
                    )
                    
                    logreg.fit(X_pca, popularities)
                    
                    # Hacer predicciones en datos de entrenamiento
                    y_pred = logreg.predict(X_pca)
                    
                    # Calcular métricas
                    accuracy = accuracy_score(popularities, y_pred)
                    precision = precision_score(popularities, y_pred, average='binary', zero_division=0)
                    recall = recall_score(popularities, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(popularities, y_pred, average='binary', zero_division=0)
                    
                    print(f"   ✅ Métricas de entrenamiento:")
                    print(f"      - Accuracy: {accuracy:.4f}")
                    print(f"      - Precision: {precision:.4f}")
                    print(f"      - Recall: {recall:.4f}")
                    print(f"      - F1-Score: {f1:.4f}")
                    
                    # Guardar modelo de regresión logística
                    logreg_model_file = f"logreg_models/cluster_{cluster_id}.pkl"
                    joblib.dump(logreg, logreg_model_file)
                    print(f"   💾 Modelo guardado: {logreg_model_file}")
                    
                    # Registrar métricas
                    training_metrics.append({
                        'cluster_id': cluster_id,
                        'n_samples': n_samples,
                        'pca_components': best_n_comp,
                        'n_populares': np.sum(popularities),
                        'n_no_populares': n_samples - np.sum(popularities),
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                    processed_clusters += 1
                    
                except Exception as e:
                    print(f"   ❌ Error procesando clúster {cluster_id}: {e}")
                    skipped_clusters += 1
                    continue
            
            # Crear DataFrame con métricas y guardarlo
            if training_metrics:
                df_metrics = pd.DataFrame(training_metrics)
                
                # Guardar métricas en CSV
                metrics_file = 'logreg_train_metrics.csv'
                df_metrics.to_csv(metrics_file, index=False)
                print(f"\n💾 Métricas guardadas en '{metrics_file}'")
                
                # Mostrar tabla resumen
                print(f"\n📊 Resumen de entrenamiento Regresión Logística:")
                print("="*95)
                print(f"{'Clúster':<8} {'Muestras':<9} {'PCA':<5} {'Pop':<4} {'NoPop':<6} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")
                print("="*95)
                
                for _, row in df_metrics.iterrows():
                    print(f"{row['cluster_id']:<8} {row['n_samples']:<9} {row['pca_components']:<5} "
                          f"{row['n_populares']:<4} {row['n_no_populares']:<6} "
                          f"{row['accuracy']:<9.4f} {row['precision']:<10.4f} "
                          f"{row['recall']:<8.4f} {row['f1_score']:<9.4f}")
                
                print("="*95)
                
                # Estadísticas globales
                print(f"\n📈 Estadísticas globales:")
                avg_accuracy = df_metrics['accuracy'].mean()
                avg_precision = df_metrics['precision'].mean()
                avg_recall = df_metrics['recall'].mean()
                avg_f1 = df_metrics['f1_score'].mean()
                
                print(f"   📊 Accuracy promedio: {avg_accuracy:.4f}")
                print(f"   📊 Precision promedio: {avg_precision:.4f}")
                print(f"   📊 Recall promedio: {avg_recall:.4f}")
                print(f"   📊 F1-Score promedio: {avg_f1:.4f}")
                
                # Mostrar distribución de componentes PCA utilizados
                pca_distribution = df_metrics['pca_components'].value_counts().sort_index()
                print(f"\n🔍 Distribución de componentes PCA utilizados:")
                for n_comp, count in pca_distribution.items():
                    print(f"   {n_comp} componentes: {count} clústeres")
                
                # Contar modelos guardados
                saved_models = [f for f in os.listdir("logreg_models") if f.startswith("cluster_") and f.endswith(".pkl")]
                print(f"\n📁 Modelos de Regresión Logística guardados: {len(saved_models)}")
                for model_file in sorted(saved_models):
                    print(f"   - {model_file}")
                
            else:
                print("❌ No se pudieron generar métricas de entrenamiento")
            
            print(f"\n✅ Proceso completado:")
            print(f"   🔢 Clústeres procesados: {processed_clusters}")
            print(f"   ⚠️  Clústeres saltados: {skipped_clusters}")
            print(f"   📁 Carpeta de modelos: logreg_models/")
            print(f"   📋 Archivo de métricas: logreg_train_metrics.csv")
            print(f"   🧠 Algoritmo: Regresión Logística (liblinear, C=1.0, L2)")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión cerrada.")

if __name__ == "__main__":
    train_logreg_per_cluster()