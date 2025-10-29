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

def evaluate_logreg_per_cluster():
    # Conectar a Neo4j
    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Verificar archivos necesarios
    if not os.path.exists("kmeans_model.pkl"):
        print("❌ Error: No se encontró kmeans_model.pkl. Ejecuta primero cluster_news.py")
        return
    
    if not os.path.exists("pca_models"):
        print("❌ Error: La carpeta 'pca_models/' no existe. Ejecuta primero reduce_pca_per_cluster.py")
        return
    
    if not os.path.exists("logreg_models"):
        print("❌ Error: La carpeta 'logreg_models/' no existe. Ejecuta primero train_logreg_per_cluster.py")
        return
    
    # Verificar archivo de métricas de entrenamiento para obtener configuraciones PCA
    if not os.path.exists("logreg_train_metrics.csv"):
        print("❌ Error: No se encontró logreg_train_metrics.csv. Ejecuta primero train_logreg_per_cluster.py")
        return
    
    try:
        # Cargar métricas de entrenamiento para obtener configuraciones PCA por clúster
        print("📊 Cargando configuraciones de entrenamiento...")
        df_train_metrics = pd.read_csv("logreg_train_metrics.csv")
        pca_config_per_cluster = dict(zip(df_train_metrics['cluster_id'], df_train_metrics['pca_components']))
        
        print("🔍 Configuraciones PCA por clúster:")
        for cluster_id, pca_comp in pca_config_per_cluster.items():
            print(f"   Clúster {cluster_id}: {pca_comp} componentes PCA")
        
        with driver.session() as session:
            # Obtener noticias de test
            print("\n📊 Obteniendo noticias de test...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'test'
                RETURN n.embedding AS embedding, n.popularity AS popularity
            """)
            
            # Recolectar datos de test
            test_embeddings = []
            test_popularities = []
            
            for record in result:
                test_embeddings.append(record["embedding"])
                test_popularities.append(record["popularity"])
            
            if not test_embeddings:
                print("❌ No se encontraron noticias de test.")
                return
            
            print(f"✅ Obtenidas {len(test_embeddings)} noticias de test")
            
            # Convertir a arrays NumPy
            X_test = np.array(test_embeddings)
            y_test = np.array(test_popularities, dtype=int)
            
            print(f"📐 Dimensiones test: {X_test.shape}")
            
            # Cargar modelo K-Means
            print("🧠 Cargando modelo K-Means...")
            with open('kmeans_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)
            
            # Predecir clusters para noticias de test
            print("🎯 Prediciendo clusters para noticias de test...")
            predicted_clusters = kmeans.predict(X_test)
            
            # Agrupar datos de test por cluster predicho
            test_clusters = {}
            for i, cluster_id in enumerate(predicted_clusters):
                if cluster_id not in test_clusters:
                    test_clusters[cluster_id] = {'embeddings': [], 'popularities': [], 'indices': []}
                
                test_clusters[cluster_id]['embeddings'].append(test_embeddings[i])
                test_clusters[cluster_id]['popularities'].append(test_popularities[i])
                test_clusters[cluster_id]['indices'].append(i)
            
            print(f"📊 Noticias agrupadas en {len(test_clusters)} clústeres")
            
            # Variables para métricas globales
            all_predictions = np.full(len(y_test), -1)  # Inicializar con -1 para detectar no procesados
            all_true_labels = y_test.copy()
            metrics_results = []
            predictions_detail = []
            processed_samples = 0
            skipped_samples = 0
            
            # Mostrar distribución por clúster
            print(f"\n📊 Distribución de muestras de test por clúster:")
            total_test = len(test_embeddings)
            for cluster_id in sorted(test_clusters.keys()):
                n_samples = len(test_clusters[cluster_id]['embeddings'])
                percentage = (n_samples / total_test) * 100
                print(f"   Clúster {cluster_id}: {n_samples} muestras ({percentage:.1f}%)")
            
            # Procesar cada clúster
            for cluster_id in sorted(test_clusters.keys()):
                try:
                    cluster_data = test_clusters[cluster_id]
                    cluster_embeddings = np.array(cluster_data['embeddings'])
                    cluster_popularities = np.array(cluster_data['popularities'], dtype=int)
                    cluster_indices = cluster_data['indices']
                    n_samples = len(cluster_embeddings)
                    
                    print(f"\n🔄 Evaluando Clúster {cluster_id} ({n_samples} muestras):")
                    
                    # Verificar configuración PCA
                    if cluster_id not in pca_config_per_cluster:
                        print(f"   ⚠️  No se encontró configuración PCA. Saltando...")
                        skipped_samples += n_samples
                        continue
                    
                    pca_components = pca_config_per_cluster[cluster_id]
                    
                    # Verificar archivos del clúster
                    pca_model_file = f"pca_models/cluster_{cluster_id}_{pca_components}.pkl"
                    logreg_model_file = f"logreg_models/cluster_{cluster_id}.pkl"
                    
                    if not os.path.exists(pca_model_file):
                        print(f"   ⚠️  No se encontró PCA: {pca_model_file}. Saltando...")
                        skipped_samples += n_samples
                        continue
                    
                    if not os.path.exists(logreg_model_file):
                        print(f"   ⚠️  No se encontró LogReg: {logreg_model_file}. Saltando...")
                        skipped_samples += n_samples
                        continue
                    
                    # Cargar modelos
                    pca_model = joblib.load(pca_model_file)
                    logreg_model = joblib.load(logreg_model_file)
                    
                    print(f"   ✅ Modelos cargados: PCA({pca_components}) + LogReg")
                    
                    # Aplicar PCA
                    X_pca = pca_model.transform(cluster_embeddings)
                    print(f"   📊 Dimensiones después de PCA: {X_pca.shape[1]}")
                    
                    # Hacer predicciones
                    cluster_predictions = logreg_model.predict(X_pca)
                    
                    # Calcular métricas del clúster
                    accuracy = accuracy_score(cluster_popularities, cluster_predictions)
                    precision = precision_score(cluster_popularities, cluster_predictions, average='binary', zero_division=0)
                    recall = recall_score(cluster_popularities, cluster_predictions, average='binary', zero_division=0)
                    f1 = f1_score(cluster_popularities, cluster_predictions, average='binary', zero_division=0)
                    
                    print(f"   ✅ Métricas de test:")
                    print(f"      - Accuracy: {accuracy:.4f}")
                    print(f"      - Precision: {precision:.4f}")
                    print(f"      - Recall: {recall:.4f}")
                    print(f"      - F1-Score: {f1:.4f}")
                    
                    # Guardar métricas del clúster
                    metrics_results.append({
                        'cluster_id': cluster_id,
                        'n_samples': n_samples,
                        'pca_features': X_pca.shape[1],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                    # Agregar predicciones a array global
                    for idx, pred in zip(cluster_indices, cluster_predictions):
                        all_predictions[idx] = pred
                    
                    # Guardar predicciones individuales
                    for i, (pred, true_val) in enumerate(zip(cluster_predictions, cluster_popularities)):
                        predictions_detail.append({
                            'cluster_id': cluster_id,
                            'sample_index': cluster_indices[i],
                            'predicted': int(pred),
                            'actual': int(true_val),
                            'correct': int(pred) == int(true_val)
                        })
                    
                    processed_samples += n_samples
                    
                except Exception as e:
                    print(f"   ❌ Error procesando clúster {cluster_id}: {e}")
                    skipped_samples += len(test_clusters[cluster_id]['embeddings'])
                    continue
            
            # Calcular métricas globales (solo con muestras procesadas)
            processed_mask = all_predictions != -1
            processed_predictions = all_predictions[processed_mask]
            processed_true_labels = all_true_labels[processed_mask]
            
            if len(processed_predictions) > 0:
                global_accuracy = accuracy_score(processed_true_labels, processed_predictions)
                global_precision = precision_score(processed_true_labels, processed_predictions, average='binary', zero_division=0)
                global_recall = recall_score(processed_true_labels, processed_predictions, average='binary', zero_division=0)
                global_f1 = f1_score(processed_true_labels, processed_predictions, average='binary', zero_division=0)
                
                # Agregar métricas globales
                metrics_results.append({
                    'cluster_id': 'GLOBAL',
                    'n_samples': len(processed_predictions),
                    'pca_features': 'N/A',
                    'accuracy': global_accuracy,
                    'precision': global_precision,
                    'recall': global_recall,
                    'f1_score': global_f1
                })
            
            # Mostrar resultados
            if metrics_results:
                df_metrics = pd.DataFrame(metrics_results)
                
                print(f"\n📊 Resultados de evaluación en test (10 clústeres, PCA adaptado):")
                print("="*75)
                print(f"{'Clúster':<8} {'Muestras':<9} {'Features':<9} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")
                print("="*75)
                
                for _, row in df_metrics.iterrows():
                    cluster_str = str(row['cluster_id'])
                    features_str = str(row['pca_features']) if row['pca_features'] != 'N/A' else 'N/A'
                    print(f"{cluster_str:<8} {row['n_samples']:<9} {features_str:<9} "
                          f"{row['accuracy']:<9.4f} {row['precision']:<10.4f} "
                          f"{row['recall']:<8.4f} {row['f1_score']:<9.4f}")
                
                print("="*75)
                
                # Guardar métricas en CSV
                df_metrics.to_csv('test_metrics_logreg_10.csv', index=False)
                print(f"📋 Métricas guardadas en 'test_metrics_logreg_10.csv'")
                
                # Guardar predicciones individuales
                if predictions_detail:
                    df_predictions = pd.DataFrame(predictions_detail)
                    df_predictions.to_csv('test_predictions_logreg_10.csv', index=False)
                    print(f"📋 Predicciones guardadas en 'test_predictions_logreg_10.csv'")
            
            # Estadísticas finales
            if len(processed_predictions) > 0:
                correct_predictions = np.sum(processed_predictions == processed_true_labels)
                total_processed = len(processed_predictions)
                success_rate = (correct_predictions / total_processed) * 100
                
                print(f"\n🎯 Resumen global:")
                print(f"   📊 Accuracy global: {global_accuracy:.4f}")
                print(f"   📊 Precision global: {global_precision:.4f}")
                print(f"   📊 Recall global: {global_recall:.4f}")
                print(f"   📊 F1-Score global: {global_f1:.4f}")
                print(f"   ✅ Predicciones correctas: {correct_predictions}/{total_processed}")
                print(f"   📈 Porcentaje de aciertos: {success_rate:.1f}%")
                
                # Distribución de clases
                true_positives = np.sum(processed_true_labels == 1)
                true_negatives = np.sum(processed_true_labels == 0)
                pred_positives = np.sum(processed_predictions == 1)
                pred_negatives = np.sum(processed_predictions == 0)
                
                print(f"\n📈 Distribución de clases:")
                print(f"   📰 Noticias populares reales: {true_positives}")
                print(f"   📰 Noticias no populares reales: {true_negatives}")
                print(f"   🎯 Predicciones populares: {pred_positives}")
                print(f"   🎯 Predicciones no populares: {pred_negatives}")
            
            # Resumen final
            total_test_samples = len(test_embeddings)
            processing_rate = (processed_samples / total_test_samples) * 100
            
            print(f"\n✅ Evaluación completada:")
            print(f"   📊 Total muestras de test: {total_test_samples}")
            print(f"   ✅ Muestras procesadas: {processed_samples}")
            print(f"   ⚠️  Muestras saltadas: {skipped_samples}")
            print(f"   📈 Tasa de procesamiento: {processing_rate:.1f}%")
            print(f"   🔢 Clústeres evaluados: {len([m for m in metrics_results if m['cluster_id'] != 'GLOBAL'])}")
            print(f"   🧠 Algoritmo: Regresión Logística + PCA adaptado por clúster")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión cerrada.")

if __name__ == "__main__":
    evaluate_logreg_per_cluster()