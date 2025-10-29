from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
import os
import pandas as pd

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def evaluate_on_test():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Verificar archivos necesarios
    if not os.path.exists("kmeans_model.pkl"):
        print("❌ Error: No se encontró kmeans_model.pkl. Ejecuta primero cluster_news.py")
        return
    
    try:
        with driver.session() as session:
            # Obtener noticias de test
            print("Obteniendo noticias de test...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'test'
                RETURN n.url AS url, n.embedding AS embedding, n.popularity AS popularity
            """)
            
            # Recolectar datos de test
            test_data = []
            for record in result:
                test_data.append({
                    'url': record["url"],
                    'embedding': record["embedding"],
                    'popularity': record["popularity"]
                })
            
            if not test_data:
                print("❌ No se encontraron noticias de test.")
                return
            
            print(f"Obtenidas {len(test_data)} noticias de test")
            
            # Cargar modelo K-Means
            print("Cargando modelo K-Means...")
            with open('kmeans_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)
            
            # Predecir clusters para noticias de test
            print("Prediciendo clusters para noticias de test...")
            test_embeddings = np.array([item['embedding'] for item in test_data])
            predicted_clusters = kmeans.predict(test_embeddings)
            
            # Agregar cluster_id predicho a los datos
            for i, cluster_id in enumerate(predicted_clusters):
                test_data[i]['predicted_cluster'] = cluster_id
            
            # Agrupar por cluster predicho
            clusters_test = {}
            for item in test_data:
                cluster_id = item['predicted_cluster']
                if cluster_id not in clusters_test:
                    clusters_test[cluster_id] = []
                clusters_test[cluster_id].append(item)
            
            print(f"Noticias agrupadas en {len(clusters_test)} clústeres")
            
            # Variables para métricas globales
            all_predictions = []
            all_true_labels = []
            metrics_results = []
            processed_news = 0
            skipped_news = 0
            
            # Procesar cada clúster
            for cluster_id in sorted(clusters_test.keys()):
                try:
                    cluster_news = clusters_test[cluster_id]
                    n_samples = len(cluster_news)
                    
                    print(f"\n🔄 Evaluando Clúster {cluster_id} ({n_samples} noticias):")
                    
                    # Verificar archivos del clúster
                    selector_path = f"reducers/selector_cluster_{cluster_id}.pkl"
                    model_path = f"models/rf_cluster_{cluster_id}.pkl"
                    
                    if not os.path.exists(selector_path):
                        print(f"⚠️  No se encontró selector: {selector_path}. Saltando...")
                        skipped_news += n_samples
                        continue
                    
                    if not os.path.exists(model_path):
                        print(f"⚠️  No se encontró modelo: {model_path}. Saltando...")
                        skipped_news += n_samples
                        continue
                    
                    # Cargar selector y modelo
                    selector = joblib.load(selector_path)
                    rf_model = joblib.load(model_path)
                    
                    # Preparar datos del clúster
                    embeddings = np.array([item['embedding'] for item in cluster_news])
                    true_labels = np.array([item['popularity'] for item in cluster_news], dtype=int)
                    urls = [item['url'] for item in cluster_news]
                    
                    # Aplicar reducción de dimensiones
                    X_reduced = selector.transform(embeddings)
                    
                    # Hacer predicciones
                    predictions = rf_model.predict(X_reduced)
                    
                    # Calcular métricas del clúster
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
                    recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
                    f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
                    
                    print(f"   ✅ Métricas de test:")
                    print(f"      - Accuracy: {accuracy:.4f}")
                    print(f"      - Precision: {precision:.4f}")
                    print(f"      - Recall: {recall:.4f}")
                    print(f"      - F1-Score: {f1:.4f}")
                    
                    # Guardar métricas del clúster
                    metrics_results.append({
                        'cluster_id': cluster_id,
                        'n_samples': n_samples,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                    # Agregar a métricas globales
                    all_predictions.extend(predictions)
                    all_true_labels.extend(true_labels)
                    
                    # Guardar predicciones individuales
                    for url, pred, true_val in zip(urls, predictions, true_labels):
                        all_predictions_detail = getattr(evaluate_on_test, 'predictions_detail', [])
                        all_predictions_detail.append({
                            'url': url,
                            'cluster_id': cluster_id,
                            'predicted': int(pred),
                            'actual': int(true_val)
                        })
                        evaluate_on_test.predictions_detail = all_predictions_detail
                    
                    processed_news += n_samples
                    
                except Exception as e:
                    print(f"❌ Error procesando clúster {cluster_id}: {e}")
                    skipped_news += len(clusters_test[cluster_id])
                    continue
            
            # Calcular métricas globales
            if all_predictions and all_true_labels:
                global_accuracy = accuracy_score(all_true_labels, all_predictions)
                global_precision = precision_score(all_true_labels, all_predictions, average='binary', zero_division=0)
                global_recall = recall_score(all_true_labels, all_predictions, average='binary', zero_division=0)
                global_f1 = f1_score(all_true_labels, all_predictions, average='binary', zero_division=0)
                
                # Agregar métricas globales a los resultados
                metrics_results.append({
                    'cluster_id': 'GLOBAL',
                    'n_samples': len(all_true_labels),
                    'accuracy': global_accuracy,
                    'precision': global_precision,
                    'recall': global_recall,
                    'f1_score': global_f1
                })
            
            # Mostrar resultados
            if metrics_results:
                df_metrics = pd.DataFrame(metrics_results)
                
                print(f"\n📊 Resultados de evaluación en test:")
                print("="*80)
                print(f"{'Clúster':<8} {'Muestras':<9} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")
                print("="*80)
                
                for _, row in df_metrics.iterrows():
                    cluster_str = str(row['cluster_id'])
                    print(f"{cluster_str:<8} {row['n_samples']:<9} "
                          f"{row['accuracy']:<9.4f} {row['precision']:<10.4f} "
                          f"{row['recall']:<8.4f} {row['f1_score']:<9.4f}")
                
                print("="*80)
                
                # Guardar métricas en CSV
                df_metrics.to_csv('test_metrics.csv', index=False)
                print(f"📋 Métricas guardadas en 'test_metrics.csv'")
                
                # Guardar predicciones individuales
                if hasattr(evaluate_on_test, 'predictions_detail'):
                    df_predictions = pd.DataFrame(evaluate_on_test.predictions_detail)
                    df_predictions.to_csv('test_predictions.csv', index=False)
                    print(f"📋 Predicciones guardadas en 'test_predictions.csv'")
            
            # Resumen final
            total_test = len(test_data)
            success_rate = (processed_news / total_test) * 100 if total_test > 0 else 0
            
            print(f"\n✅ Evaluación completada:")
            print(f"   Total noticias de test: {total_test}")
            print(f"   Noticias procesadas exitosamente: {processed_news}")
            print(f"   Noticias saltadas: {skipped_news}")
            print(f"   Tasa de éxito: {success_rate:.1f}%")
            
            # Distribución por clúster
            print(f"\n📊 Distribución por clúster:")
            for cluster_id in sorted(clusters_test.keys()):
                n_samples = len(clusters_test[cluster_id])
                percentage = (n_samples / total_test) * 100
                print(f"   Clúster {cluster_id}: {n_samples} noticias ({percentage:.1f}%)")
            
            # Mostrar métricas globales finales
            if all_predictions and all_true_labels:
                print(f"\n🎯 Métricas globales en test:")
                print(f"   - Accuracy global: {global_accuracy:.4f}")
                print(f"   - Precision global: {global_precision:.4f}")
                print(f"   - Recall global: {global_recall:.4f}")
                print(f"   - F1-Score global: {global_f1:.4f}")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\nConexión cerrada.")

if __name__ == "__main__":
    evaluate_on_test()