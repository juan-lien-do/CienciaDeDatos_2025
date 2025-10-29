from neo4j import GraphDatabase
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import pandas as pd

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def train_random_forests_5_40():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Crear carpeta models_5_40 si no existe
    os.makedirs("models_5_40", exist_ok=True)
    print("Carpeta 'models_5_40/' creada o ya existe")
    
    # Verificar que exista la carpeta reducers_40
    if not os.path.exists("reducers_40"):
        print("❌ Error: La carpeta 'reducers_40/' no existe. Ejecuta primero reduce_dimensions_40.py")
        return
    
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
            
            # Lista para almacenar métricas
            metrics_results = []
            processed_clusters = 0
            skipped_clusters = 0
            
            # Procesar cada clúster
            for cluster_id in sorted(clusters_data.keys()):
                try:
                    print(f"\n🔄 Procesando Clúster {cluster_id}:")
                    
                    embeddings = np.array(clusters_data[cluster_id]["embeddings"])
                    popularities = np.array(clusters_data[cluster_id]["popularities"], dtype=int)
                    
                    n_samples = len(embeddings)
                    print(f"   - Noticias: {n_samples}")
                    
                    # Verificar tamaño mínimo del clúster
                    if n_samples < 10:
                        print(f"⚠️  Clúster {cluster_id}: Solo {n_samples} noticias (mínimo 10). Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Verificar que hay variabilidad en las etiquetas
                    unique_labels = np.unique(popularities)
                    if len(unique_labels) < 2:
                        print(f"⚠️  Clúster {cluster_id}: Solo una clase de popularidad. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    # Cargar el selector correspondiente desde reducers_40
                    selector_path = f"reducers_40/selector_cluster_{cluster_id}.pkl"
                    if not os.path.exists(selector_path):
                        print(f"⚠️  Clúster {cluster_id}: No se encontró el selector {selector_path}. Saltando...")
                        skipped_clusters += 1
                        continue
                    
                    selector = joblib.load(selector_path)
                    print(f"   - Selector cargado: {selector_path}")
                    
                    # Aplicar reducción de dimensiones (40 features)
                    X_reduced = selector.transform(embeddings)
                    print(f"   - Dimensiones reducidas: {X_reduced.shape[1]} features")
                    
                    # Entrenar RandomForestClassifier
                    rf = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        max_depth=None,
                        n_jobs=-1  # Usar todos los cores disponibles
                    )
                    
                    rf.fit(X_reduced, popularities)
                    
                    # Calcular métricas en datos de entrenamiento
                    y_pred = rf.predict(X_reduced)
                    
                    accuracy = accuracy_score(popularities, y_pred)
                    precision = precision_score(popularities, y_pred, average='binary', zero_division=0)
                    recall = recall_score(popularities, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(popularities, y_pred, average='binary', zero_division=0)
                    
                    print(f"   ✅ Métricas de entrenamiento:")
                    print(f"      - Accuracy: {accuracy:.4f}")
                    print(f"      - Precision: {precision:.4f}")
                    print(f"      - Recall: {recall:.4f}")
                    print(f"      - F1-Score: {f1:.4f}")
                    
                    # Guardar el modelo en models_5_40
                    model_path = f"models_5_40/rf_cluster_{cluster_id}.pkl"
                    joblib.dump(rf, model_path)
                    print(f"   💾 Modelo guardado: {model_path}")
                    
                    # Registrar métricas
                    metrics_results.append({
                        'cluster_id': cluster_id,
                        'n_samples': n_samples,
                        'n_features': X_reduced.shape[1],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                    processed_clusters += 1
                    
                except Exception as e:
                    print(f"❌ Error procesando clúster {cluster_id}: {e}")
                    skipped_clusters += 1
                    continue
            
            # Crear DataFrame con resultados
            if metrics_results:
                df_metrics = pd.DataFrame(metrics_results)
                
                print(f"\n📊 Resumen de entrenamiento (5 clústeres, 40 features):")
                print("="*80)
                print(f"{'Clúster':<8} {'Muestras':<9} {'Features':<9} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")
                print("="*80)
                
                for _, row in df_metrics.iterrows():
                    print(f"{row['cluster_id']:<8} {row['n_samples']:<9} {row['n_features']:<9} "
                          f"{row['accuracy']:<9.4f} {row['precision']:<10.4f} {row['recall']:<8.4f} {row['f1_score']:<9.4f}")
                
                print("="*80)
                
                # Métricas promedio
                mean_metrics = df_metrics[['accuracy', 'precision', 'recall', 'f1_score']].mean()
                print(f"\n🎯 Métricas promedio globales:")
                print(f"   - Accuracy promedio: {mean_metrics['accuracy']:.4f}")
                print(f"   - Precision promedio: {mean_metrics['precision']:.4f}")
                print(f"   - Recall promedio: {mean_metrics['recall']:.4f}")
                print(f"   - F1-Score promedio: {mean_metrics['f1_score']:.4f}")
                
                # Guardar métricas en CSV
                df_metrics.to_csv('training_metrics_5_40.csv', index=False)
                print(f"   📋 Métricas guardadas en 'training_metrics_5_40.csv'")
            
            # Resumen final
            print(f"\n✅ Proceso de entrenamiento completado (5 clústeres, 40 features):")
            print(f"   Total de clústeres encontrados: {len(clusters_data)}")
            print(f"   Clústeres entrenados exitosamente: {processed_clusters}")
            print(f"   Clústeres saltados: {skipped_clusters}")
            
            # Verificar archivos guardados en models_5_40
            saved_models = [f for f in os.listdir("models_5_40") if f.startswith("rf_cluster_")]
            print(f"   Modelos guardados en 'models_5_40/': {len(saved_models)}")
            
            if saved_models:
                print("   📁 Modelos creados en models_5_40/:")
                for model in sorted(saved_models):
                    print(f"      - {model}")
            
            # Mostrar comparación con configuración anterior
            print(f"\n📈 Configuración del modelo:")
            print(f"   - Clústeres: 5 (vs 10 anterior)")
            print(f"   - Features por clúster: 40 (vs 20 anterior)")
            print(f"   - Random Forest estimadores: 200")
            print(f"   - Carpeta de salida: models_5_40/")
            print(f"   - Métricas guardadas: training_metrics_5_40.csv")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\nConexión cerrada.")

if __name__ == "__main__":
    train_random_forests_5_40()