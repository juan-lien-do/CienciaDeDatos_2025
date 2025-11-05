from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from umap import UMAP
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configuración del análisis
UMAP_COMPONENTS = 20  # Número de componentes para UMAP
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLUSTER = 50  # Mínimo de muestras para entrenar un modelo (aumentado para CV)
CV_FOLDS = 5  # Número de folds para validación cruzada


def analyze_viral_prediction_by_cluster():
    """
    Analiza la capacidad de predecir viralidad por clúster usando embeddings de títulos
    con UMAP + validación cruzada estratificada.
    
    Flujo:
    1. Carga datos desde Neo4j (subset='train')
    2. Para cada cluster_id: UMAP + XGBoost + validación cruzada (5 folds)
    3. Calcula métricas promedio: Accuracy, F1-score, ROC AUC
    4. Genera tabla resumen con importancia de features
    5. Muestra advertencias para clústeres con pocas muestras
    """
    
    print("🚀 Iniciando análisis de predicción viral por clúster")
    print("=" * 70)
    
    # 1️⃣ Cargar datos desde Neo4j
    print("📊 Cargando datos desde Neo4j...")
    df = load_data_from_neo4j()
    
    if df is None or len(df) == 0:
        print("❌ No se pudieron cargar datos. Verifica la conexión y que existan datos.")
        return
    
    print(f"✅ Datos cargados: {len(df)} registros (subset='train')")
    print(f"📊 Columnas disponibles: {list(df.columns)}")
    print(f"🎯 Clústeres únicos (cluster_id): {df['cluster'].nunique()}")
    print(f"📈 Distribución viral (popularity): {df['viral'].value_counts().to_dict()}")
    print(f"📏 Dimensión embeddings: {len(df['embedding_titulo'].iloc[0]) if len(df) > 0 else 'N/A'}")
    
    # 2️⃣ Analizar cada clúster individualmente
    print(f"\n🔍 Analizando predicción viral por clúster...")
    cluster_results = analyze_clusters(df)
    
    # 3️⃣ Mostrar tabla resumen
    print_summary_table(cluster_results)
    
    # 4️⃣ Mostrar tabla con pandas DataFrame
    show_results_dataframe(cluster_results)
    
    # 5️⃣ Generar gráficos comparativos
    plot_cluster_performance(cluster_results)
    
    # 6️⃣ Destacar mejores y peores clústeres
    highlight_best_worst_clusters(cluster_results)
    
    print(f"\n✅ Análisis completado. Gráfico guardado como 'cluster_performance_umap_xgb.png'")


def load_data_from_neo4j():
    """
    Carga datos de entrenamiento desde Neo4j con título, viralidad, clúster y embeddings.
    
    Returns:
        pandas.DataFrame: DataFrame con datos listos para análisis
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Query para obtener datos de entrenamiento con todos los campos necesarios
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train' 
                  AND n.embedding_titulo IS NOT NULL 
                  AND n.cluster_id IS NOT NULL
                  AND n.popularity IS NOT NULL
                RETURN 
                    n.titulo AS titulo,
                    n.popularity AS viral,
                    n.cluster_id AS cluster,
                    n.embedding_titulo AS embedding_titulo
            """)
            
            # Convertir a lista de diccionarios
            data = []
            for record in result:
                data.append({
                    'titulo': record['titulo'],
                    'viral': int(record['viral']),  # Convertir boolean a int (0/1)
                    'cluster': int(record['cluster']),
                    'embedding_titulo': record['embedding_titulo']
                })
            
            if not data:
                return None
            
            # Crear DataFrame
            df = pd.DataFrame(data)
            
            # Validar que tenemos embeddings válidos
            df = df[df['embedding_titulo'].apply(lambda x: x is not None and len(x) > 0)]
            
            return df
            
    except Exception as e:
        print(f"❌ Error cargando datos desde Neo4j: {e}")
        return None
    
    finally:
        driver.close()


def analyze_clusters(df):
    """
    Analiza cada clúster individualmente aplicando UMAP + XGBoost.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de entrenamiento
        
    Returns:
        list: Lista de resultados por clúster
    """
    cluster_results = []
    clusters = sorted(df['cluster'].unique())
    
    print(f"\n🧠 Procesando {len(clusters)} clústeres...")
    
    for cluster_id in clusters:
        try:
            print(f"\n🔄 Procesando Clúster {cluster_id}:")
            
            # Filtrar registros del clúster
            cluster_data = df[df['cluster'] == cluster_id].copy()
            n_samples = len(cluster_data)
            
            print(f"   📊 Muestras en clúster: {n_samples}")
            
            # Verificar tamaño mínimo del clúster (advertencia para <50 muestras)
            if n_samples < MIN_SAMPLES_PER_CLUSTER:
                print(f"   ⚠️  ADVERTENCIA: Pocas muestras para validación cruzada robusta "
                      f"(tiene {n_samples}, recomendado ≥{MIN_SAMPLES_PER_CLUSTER})")
            
            # Mínimo absoluto para poder hacer CV de 5 folds
            if n_samples < 10:
                print(f"   ❌ Muy pocas muestras para CV (mínimo 10). Saltando...")
                continue
            
            # Verificar que hay variabilidad en la variable objetivo
            viral_counts = cluster_data['viral'].value_counts()
            if len(viral_counts) < 2:
                print(f"   ⚠️  Solo una clase de viralidad. Saltando...")
                continue
            
            print(f"   📈 Distribución viral: {viral_counts.to_dict()}")
            
            # Preparar embeddings
            embeddings = np.array(cluster_data['embedding_titulo'].tolist())
            y = cluster_data['viral'].values
            
            print(f"   📐 Dimensiones embeddings: {embeddings.shape}")
            
            # Aplicar UMAP
            n_components = min(UMAP_COMPONENTS, n_samples - 1, embeddings.shape[1])
            print(f"   🔄 Aplicando UMAP con {n_components} componentes...")
            
            umap_reducer = UMAP(
                n_components=n_components,
                random_state=RANDOM_STATE,
                n_neighbors=min(15, n_samples - 1),  # Ajustar neighbors si hay pocas muestras
                min_dist=0.1,
                metric='cosine'  # Bueno para embeddings de texto
            )
            
            X_umap = umap_reducer.fit_transform(embeddings)
            
            print(f"   🗺️  Componentes UMAP: {n_components}")
            print(f"   📊 Forma después de UMAP: {X_umap.shape}")
            
            # Configurar XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0  # Silenciar warnings de XGBoost
            )
            
            # Configurar validación cruzada estratificada
            cv_folds = min(CV_FOLDS, n_samples // 2)  # Ajustar folds si hay pocas muestras
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            
            print(f"   🔄 Realizando validación cruzada ({cv_folds} folds)...")
            
            # Calcular métricas con validación cruzada
            # Accuracy
            cv_accuracy = cross_val_score(xgb_model, X_umap, y, cv=skf, scoring='accuracy', n_jobs=1)  # XGBoost maneja paralelismo internamente
            accuracy_mean = cv_accuracy.mean()
            accuracy_std = cv_accuracy.std()
            
            # F1-Score
            f1_scorer = make_scorer(f1_score, zero_division=0)
            cv_f1 = cross_val_score(xgb_model, X_umap, y, cv=skf, scoring=f1_scorer, n_jobs=1)
            f1_mean = cv_f1.mean()
            f1_std = cv_f1.std()
            
            # ROC AUC (solo si hay ambas clases)
            try:
                cv_auc = cross_val_score(xgb_model, X_umap, y, cv=skf, scoring='roc_auc', n_jobs=1)
                auc_mean = cv_auc.mean()
                auc_std = cv_auc.std()
            except Exception:
                auc_mean = 0.5  # AUC neutral si no se puede calcular
                auc_std = 0.0
            
            print(f"   ✅ Métricas (CV promedio ± std):")
            print(f"      - Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
            print(f"      - F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
            print(f"      - ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}")
            
            # Entrenar modelo completo para obtener importancia de features
            xgb_model.fit(X_umap, y)
            feature_importance = xgb_model.feature_importances_
            top5_features = np.argsort(feature_importance)[-5:][::-1]  # Top 5 descendente
            avg_importance_top5 = np.mean(feature_importance[top5_features])
            
            print(f"   🏆 Importancia promedio top 5 features (XGBoost): {avg_importance_top5:.4f}")
            print(f"      Top 5 componentes UMAP: {top5_features.tolist()}")
            
            # Guardar resultados
            cluster_results.append({
                'cluster_id': cluster_id,
                'n_samples': n_samples,
                'n_components': n_components,
                'umap_shape': X_umap.shape,
                'accuracy': accuracy_mean,
                'accuracy_std': accuracy_std,
                'f1_score': f1_mean,
                'f1_std': f1_std,
                'roc_auc': auc_mean,
                'roc_auc_std': auc_std,
                'avg_importance_top5': avg_importance_top5,
                'top5_features': top5_features.tolist(),
                'viral_distribution': viral_counts.to_dict(),
                'cv_folds': cv_folds
            })
            
        except Exception as e:
            print(f"   ❌ Error procesando clúster {cluster_id}: {e}")
            continue
    
    return cluster_results


def print_summary_table(results):
    """
    Imprime tabla resumen con métricas por clúster.
    
    Args:
        results (list): Lista de resultados por clúster
    """
    if not results:
        print("❌ No hay resultados para mostrar.")
        return
    
    print(f"\n📊 TABLA RESUMEN - PREDICCIÓN VIRAL POR CLÚSTER (UMAP + XGBoost + Validación Cruzada)")
    print("=" * 115)
    print(f"{'Clúster':<8} {'Muestras':<9} {'UMAP':<5} {'Accuracy':<15} {'F1-Score':<15} {'ROC AUC':<15} {'Imp.Top5':<9}")
    print(f"{'':^8} {'':^9} {'':^5} {'(promedio±std)':<15} {'(promedio±std)':<15} {'(promedio±std)':<15} {'(XGBoost)':<9}")
    print("=" * 115)
    
    total_samples = 0
    total_accuracy = 0
    total_f1 = 0
    total_auc = 0
    
    for result in results:
        cluster_id = result['cluster_id']
        n_samples = result['n_samples']
        n_components = result['n_components']
        accuracy = result['accuracy']
        accuracy_std = result['accuracy_std']
        f1 = result['f1_score']
        f1_std = result['f1_std']
        auc = result['roc_auc']
        auc_std = result['roc_auc_std']
        importance = result['avg_importance_top5']
        
        print(f"{cluster_id:<8} {n_samples:<9} {n_components:<5} "
              f"{accuracy:.3f}±{accuracy_std:.3f}    "
              f"{f1:.3f}±{f1_std:.3f}    "
              f"{auc:.3f}±{auc_std:.3f}    "
              f"{importance:<9.4f}")
        
        # Acumular para promedio ponderado
        total_samples += n_samples
        total_accuracy += accuracy * n_samples
        total_f1 += f1 * n_samples
        total_auc += auc * n_samples
    
    print("=" * 115)
    
    # Mostrar promedios ponderados
    if total_samples > 0:
        avg_accuracy = total_accuracy / total_samples
        avg_f1 = total_f1 / total_samples
        avg_auc = total_auc / total_samples
        
        print(f"{'PROMEDIO':<8} {total_samples:<9} {'N/A':<5} "
              f"{avg_accuracy:.4f}        "
              f"{avg_f1:.4f}        "
              f"{avg_auc:.4f}        "
              f"{'N/A':<9}")
        print("=" * 115)


def show_results_dataframe(results):
    """
    Muestra los resultados en una tabla usando pandas DataFrame.
    
    Args:
        results (list): Lista de resultados por clúster
    """
    if not results:
        print("❌ No hay resultados para mostrar en DataFrame.")
        return
    
    # Crear DataFrame con los resultados
    df_data = []
    for result in results:
        df_data.append({
            'Cluster': result['cluster_id'],
            'Muestras': result['n_samples'],
            'Dim_reduccion': f"UMAP-{result['n_components']}",
            'Accuracy': f"{result['accuracy']:.3f}±{result['accuracy_std']:.3f}",
            'F1_Score': f"{result['f1_score']:.3f}±{result['f1_std']:.3f}",
            'ROC_AUC': f"{result['roc_auc']:.3f}±{result['roc_auc_std']:.3f}",
            'Importancia_Top5': f"{result['avg_importance_top5']:.4f}"
        })
    
    df_results = pd.DataFrame(df_data)
    
    print(f"\n📋 TABLA RESUMEN - ANÁLISIS CON UMAP + XGBoost (DataFrame)")
    print("=" * 85)
    print(df_results.to_string(index=False))
    print("=" * 85)
    
    # Calcular estadísticas generales
    total_muestras = sum(r['n_samples'] for r in results)
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])
    avg_auc = np.mean([r['roc_auc'] for r in results])
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   🎯 Clústeres analizados: {len(results)}")
    print(f"   📊 Total de muestras: {total_muestras}")
    print(f"   📈 Accuracy promedio: {avg_accuracy:.4f}")
    print(f"   📈 F1-Score promedio: {avg_f1:.4f}")
    print(f"   📈 ROC AUC promedio: {avg_auc:.4f}")


def plot_cluster_performance(results):
    """
    Genera gráfico de barras comparando rendimiento por clúster.
    
    Args:
        results (list): Lista de resultados por clúster
    """
    if not results:
        print("❌ No hay datos para graficar.")
        return
    
    # Extraer datos para gráficos
    cluster_ids = [r['cluster_id'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    
    # Configurar gráfico
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gráfico 1: Accuracy por clúster
    bars1 = ax1.bar(cluster_ids, accuracies, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Clúster ID')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy por Clúster')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Destacar mejor accuracy
    best_acc_idx = np.argmax(accuracies)
    bars1[best_acc_idx].set_color('gold')
    
    # Gráfico 2: F1-Score por clúster
    bars2 = ax2.bar(cluster_ids, f1_scores, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Clúster ID')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score por Clúster')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Destacar mejor F1
    best_f1_idx = np.argmax(f1_scores)
    bars2[best_f1_idx].set_color('gold')
    
    # Gráfico 3: ROC AUC por clúster
    bars3 = ax3.bar(cluster_ids, roc_aucs, color='coral', alpha=0.7)
    ax3.set_xlabel('Clúster ID')
    ax3.set_ylabel('ROC AUC')
    ax3.set_title('ROC AUC por Clúster')
    ax3.set_ylim(0.4, 1)  # AUC desde 0.4 para mejor visualización
    ax3.grid(True, alpha=0.3)
    
    # Destacar mejor AUC
    best_auc_idx = np.argmax(roc_aucs)
    bars3[best_auc_idx].set_color('gold')
    
    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig('cluster_performance_umap_xgb.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Gráfico guardado como 'cluster_performance_umap_xgb.png'")


def highlight_best_worst_clusters(results):
    """
    Destaca los mejores y peores clústeres para predicción viral.
    
    Args:
        results (list): Lista de resultados por clúster
    """
    if not results:
        return
    
    # Ordenar por F1-Score (métrica más balanceada para clasificación)
    results_sorted = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n🏆 ANÁLISIS DE CLÚSTERES PARA PREDICCIÓN VIRAL")
    print("=" * 60)
    
    # Mejores clústeres (top 3)
    print(f"🥇 MEJORES CLÚSTERES (Top 3 por F1-Score):")
    for i, result in enumerate(results_sorted[:3], 1):
        cluster_id = result['cluster_id']
        f1 = result['f1_score']
        accuracy = result['accuracy']
        auc = result['roc_auc']
        n_samples = result['n_samples']
        
        print(f"   {i}. Clúster {cluster_id}: F1={f1:.4f}, Acc={accuracy:.4f}, "
              f"AUC={auc:.4f} ({n_samples} muestras)")
    
    # Peores clústeres (bottom 3)
    print(f"\n🔻 CLÚSTERES CON MENOR RENDIMIENTO (Bottom 3 por F1-Score):")
    for i, result in enumerate(results_sorted[-3:], 1):
        cluster_id = result['cluster_id']
        f1 = result['f1_score']
        accuracy = result['accuracy']
        auc = result['roc_auc']
        n_samples = result['n_samples']
        
        print(f"   {i}. Clúster {cluster_id}: F1={f1:.4f}, Acc={accuracy:.4f}, "
              f"AUC={auc:.4f} ({n_samples} muestras)")
    
    # Estadísticas generales
    f1_scores = [r['f1_score'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    aucs = [r['roc_auc'] for r in results]
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   📊 F1-Score promedio: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"   📊 Accuracy promedio: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"   📊 ROC AUC promedio: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
    print(f"   🎯 Clústeres analizados: {len(results)}")
    
    # Interpretación
    best_cluster = results_sorted[0]
    worst_cluster = results_sorted[-1]
    
    print(f"\n💡 INTERPRETACIÓN:")
    print(f"   🏆 El clúster {best_cluster['cluster_id']} es el más predictivo "
          f"(F1={best_cluster['f1_score']:.4f})")
    print(f"   📉 El clúster {worst_cluster['cluster_id']} es el menos predictivo "
          f"(F1={worst_cluster['f1_score']:.4f})")
    
    f1_range = max(f1_scores) - min(f1_scores)
    print(f"   📊 Rango de rendimiento: {f1_range:.4f} (diferencia máx-mín F1)")
    
    if f1_range > 0.2:
        print(f"   ✨ Hay variabilidad significativa entre clústeres - "
              f"algunos temas predicen mejor viralidad")
    else:
        print(f"   📊 Rendimiento relativamente uniforme entre clústeres")


if __name__ == "__main__":
    print("🚀 Iniciando análisis de predicción viral por clúster con UMAP + XGBoost + validación cruzada")
    print(f"📊 Configuración: UMAP={UMAP_COMPONENTS} componentes, XGBoost=100 estimadores, CV={CV_FOLDS} folds")
    print(f"🗺️  Reducción dimensionalidad: UMAP (preserva estructura local en embeddings)")
    print(f"🚀 Clasificador: XGBoost (gradient boosting optimizado)")
    print(f"🎯 Objetivo: Evaluar qué clústeres de títulos predicen mejor la viralidad (UMAP+XGBoost)")
    print(f"⚠️  Advertencia si clúster tiene <{MIN_SAMPLES_PER_CLUSTER} muestras")
    print("=" * 90)
    
    analyze_viral_prediction_by_cluster()