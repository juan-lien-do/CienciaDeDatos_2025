from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def analyze_emotions_by_cluster():
    """
    Analiza la capacidad predictiva de emociones por cada clúster de noticias.
    
    Flujo:
    1. Carga datos de entrenamiento con clústeres y emociones
    2. Para cada clúster (k=16), entrena modelo XGBoost independiente
    3. Evalúa rendimiento con validación cruzada por clúster
    4. Compara métricas entre clústeres para identificar patrones
    5. Visualiza resultados y rankings de clústeres por rendimiento
    """
    
    print("🎯 ANÁLISIS DE EMOCIONES POR CLÚSTER (K-MEANS k=16)")
    print("=" * 80)
    print("🧠 Objetivo: Evaluar si las emociones predicen mejor en ciertos clústeres")
    print("🤖 Modelo: XGBoost con validación cruzada por clúster")
    print("📊 Métricas: F1, Precision, Recall, ROC-AUC por clúster")
    print("🎭 Hipótesis: Diferentes tipos de contenido responden distinto a emociones")
    print("=" * 80)
    
    # Cargar datos con clústeres
    df_train = load_clustered_data()
    
    if df_train.empty:
        print("❌ No se encontraron datos con clústeres")
        return
    
    # Analizar distribución de clústeres
    analyze_cluster_distribution(df_train)
    
    # Evaluar por cada clúster
    cluster_results = {}
    
    for cluster_id in range(16):
        print(f"\n🔍 PROCESANDO CLÚSTER {cluster_id}")
        print("-" * 50)
        
        cluster_data = df_train[df_train['cluster_id'] == cluster_id]
        
        if len(cluster_data) < 50:  # Mínimo para análisis válido
            print(f"   ⚠️ Clúster {cluster_id}: Solo {len(cluster_data)} muestras, saltando...")
            continue
            
        # Preparar features y evaluar modelo para este clúster
        results = evaluate_cluster_emotions(cluster_data, cluster_id)
        if results:
            cluster_results[cluster_id] = results
    
    # Comparar resultados entre clústeres
    compare_cluster_performance(cluster_results, df_train)
    
    # Análisis de features importantes por clúster
    analyze_cluster_feature_patterns(cluster_results)
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS POR CLÚSTER COMPLETADO")
    print("📊 Revisa los gráficos y tablas generadas")
    print("💡 Identifica clústeres donde las emociones son más predictivas")
    print("=" * 80)


def load_clustered_data():
    """
    Carga datos de entrenamiento con información de clústeres y emociones.
    
    Returns:
        pd.DataFrame: Datos con clúster, emociones y viralidad
    """
    print("\n📊 Cargando datos con clústeres...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            query = """
                MATCH (n:Noticia)
                WHERE n.subset = 'train'
                  AND n.cluster_id IS NOT NULL
                  AND n.popularity IS NOT NULL
                  AND n.analisis_sentimiento_titulo_label IS NOT NULL
                  AND n.analisis_emocion_titulo_label IS NOT NULL
                RETURN 
                    n.titulo AS titulo,
                    n.shares AS shares,
                    n.popularity AS is_viral,
                    n.cluster_id AS cluster_id,
                    n.analisis_sentimiento_titulo_label AS sentiment_label,
                    n.analisis_sentimiento_titulo_score AS sentiment_score,
                    n.analisis_sentimiento_titulo_all_labels AS sentiment_all_labels,
                    n.analisis_sentimiento_titulo_all_scores AS sentiment_all_scores,
                    n.analisis_emocion_titulo_label AS emotion_label,
                    n.analisis_emocion_titulo_score AS emotion_score,
                    n.analisis_emocion_titulo_all_labels AS emotion_all_labels,
                    n.analisis_emocion_titulo_all_scores AS emotion_all_scores
            """
            
            result = session.run(query)
            records = [dict(record) for record in result]
            
            if not records:
                print("   ❌ No se encontraron datos con análisis completo y clústeres")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            print(f"   ✅ Cargados {len(df)} registros con clústeres")
            print(f"   📈 Clústeres únicos: {sorted(df['cluster_id'].unique())}")
            print(f"   🎯 Distribución de viralidad: {df['is_viral'].value_counts().to_dict()}")
            
            return df
            
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return pd.DataFrame()
        
    finally:
        driver.close()


def analyze_cluster_distribution(df):
    """
    Analiza la distribución de muestras y viralidad por clúster.
    
    Args:
        df (pd.DataFrame): Datos de entrenamiento
    """
    print("\n📊 DISTRIBUCIÓN POR CLÚSTER:")
    print("-" * 50)
    
    cluster_stats = []
    
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_data = df[df['cluster_id'] == cluster_id]
        total = len(cluster_data)
        viral_count = cluster_data['is_viral'].sum()
        viral_pct = (viral_count / total) * 100 if total > 0 else 0
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'total_samples': total,
            'viral_count': viral_count,
            'viral_percentage': viral_pct,
            'non_viral_count': total - viral_count
        })
        
        print(f"   Clúster {cluster_id:2d}: {total:5d} muestras | "
              f"Virales: {viral_count:4d} ({viral_pct:5.1f}%) | "
              f"No virales: {total-viral_count:4d}")
    
    return cluster_stats


def prepare_cluster_features(cluster_data):
    """
    Prepara features de emociones para un clúster específico.
    
    Args:
        cluster_data (pd.DataFrame): Datos del clúster
        
    Returns:
        tuple: (X, y, feature_names)
    """
    # Resetear índices para evitar problemas de alineación
    cluster_data_reset = cluster_data.reset_index(drop=True)
    
    # Crear DataFrame para features
    features_df = pd.DataFrame()
    
    # === ONE-HOT ENCODING PARA EMOCIONES ===
    emotion_dummies = pd.get_dummies(cluster_data_reset['emotion_label'], prefix='emotion')
    features_df = pd.concat([features_df, emotion_dummies], axis=1)
    
    # === ONE-HOT ENCODING PARA SENTIMIENTOS ===
    sentiment_dummies = pd.get_dummies(cluster_data_reset['sentiment_label'], prefix='sentiment')
    features_df = pd.concat([features_df, sentiment_dummies], axis=1)
    
    # === SCORES NUMÉRICAS ===
    features_df['emotion_confidence'] = cluster_data_reset['emotion_score'].values
    features_df['sentiment_confidence'] = cluster_data_reset['sentiment_score'].values
    
    # === FEATURES DERIVADAS ===
    # Procesar arrays de emociones y sentimientos específicos
    for idx, row in cluster_data_reset.iterrows():
        # Emociones específicas
        if isinstance(row['emotion_all_labels'], list) and isinstance(row['emotion_all_scores'], list):
            emotion_dict = dict(zip(row['emotion_all_labels'], row['emotion_all_scores']))
            
            for emotion in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'optimism']:
                score = emotion_dict.get(emotion, 0.0)
                if f'emotion_score_{emotion}' not in features_df.columns:
                    features_df[f'emotion_score_{emotion}'] = 0.0
                features_df.at[idx, f'emotion_score_{emotion}'] = score
        
        # Sentimientos específicos
        if isinstance(row['sentiment_all_labels'], list) and isinstance(row['sentiment_all_scores'], list):
            sentiment_dict = dict(zip(row['sentiment_all_labels'], row['sentiment_all_scores']))
            
            for sentiment in ['negative', 'neutral', 'positive']:
                score = sentiment_dict.get(sentiment, 0.0)
                if f'sentiment_score_{sentiment}' not in features_df.columns:
                    features_df[f'sentiment_score_{sentiment}'] = 0.0
                features_df.at[idx, f'sentiment_score_{sentiment}'] = score
    
    # === ENTROPÍA EMOCIONAL ===
    emotion_entropy = []
    sentiment_entropy = []
    
    for idx, row in cluster_data_reset.iterrows():
        # Entropía de emociones
        if isinstance(row['emotion_all_scores'], list):
            scores = np.array(row['emotion_all_scores'])
            scores = scores / (scores.sum() + 1e-10)
            entropy = -np.sum(scores * np.log(scores + 1e-10))
            emotion_entropy.append(entropy)
        else:
            emotion_entropy.append(0.0)
            
        # Entropía de sentimientos
        if isinstance(row['sentiment_all_scores'], list):
            scores = np.array(row['sentiment_all_scores'])
            scores = scores / (scores.sum() + 1e-10)
            entropy = -np.sum(scores * np.log(scores + 1e-10))
            sentiment_entropy.append(entropy)
        else:
            sentiment_entropy.append(0.0)
    
    features_df['emotion_entropy'] = emotion_entropy
    features_df['sentiment_entropy'] = sentiment_entropy
    
    # TARGET
    y = cluster_data_reset['is_viral'].astype(int)
    
    # Rellenar valores faltantes
    features_df = features_df.fillna(0)
    
    return features_df.values, y.values, features_df.columns.tolist()


def evaluate_cluster_emotions(cluster_data, cluster_id):
    """
    Evalúa la capacidad predictiva de emociones para un clúster específico.
    
    Args:
        cluster_data (pd.DataFrame): Datos del clúster
        cluster_id (int): ID del clúster
        
    Returns:
        dict: Resultados de evaluación del clúster
    """
    print(f"   📊 Muestras: {len(cluster_data)}")
    viral_count = cluster_data['is_viral'].sum()
    print(f"   🔥 Virales: {viral_count} ({viral_count/len(cluster_data)*100:.1f}%)")
    
    # Verificar que hay suficientes muestras de ambas clases
    if viral_count == 0 or viral_count == len(cluster_data):
        print(f"   ❌ Clúster {cluster_id}: Solo una clase presente, saltando...")
        return None
    
    if min(viral_count, len(cluster_data) - viral_count) < 10:
        print(f"   ❌ Clúster {cluster_id}: Muy pocas muestras de una clase, saltando...")
        return None
    
    # Preparar features
    try:
        X, y, feature_names = prepare_cluster_features(cluster_data)
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            print(f"   ❌ Clúster {cluster_id}: Features vacías")
            return None
            
    except Exception as e:
        print(f"   ❌ Error preparando features para clúster {cluster_id}: {e}")
        return None
    
    # Configurar modelo XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Validación cruzada estratificada
    try:
        cv_folds = min(5, min(viral_count, len(cluster_data) - viral_count))  # Ajustar folds si es necesario
        if cv_folds < 2:
            print(f"   ❌ Clúster {cluster_id}: No suficientes muestras para CV")
            return None
            
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Métricas de validación cruzada
        cv_f1 = cross_val_score(xgb_model, X, y, cv=cv, scoring='f1')
        cv_precision = cross_val_score(xgb_model, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(xgb_model, X, y, cv=cv, scoring='recall')
        cv_accuracy = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
        cv_roc_auc = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
        
        # Entrenar modelo final para feature importance
        xgb_model.fit(X, y)
        feature_importance = dict(zip(feature_names, xgb_model.feature_importances_))
        
        # Mostrar resultados
        print(f"   🎯 F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"   🎯 Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        print(f"   🎯 Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        print(f"   🎯 ROC-AUC:   {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
        
        return {
            'cluster_id': cluster_id,
            'n_samples': len(cluster_data),
            'n_viral': viral_count,
            'viral_rate': viral_count / len(cluster_data),
            'cv_folds': cv_folds,
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'roc_auc_mean': cv_roc_auc.mean(),
            'roc_auc_std': cv_roc_auc.std(),
            'feature_importance': feature_importance,
            'top_emotions': cluster_data['emotion_label'].value_counts().head(3).to_dict(),
            'top_sentiments': cluster_data['sentiment_label'].value_counts().head(3).to_dict()
        }
        
    except Exception as e:
        print(f"   ❌ Error en validación cruzada para clúster {cluster_id}: {e}")
        return None


def compare_cluster_performance(cluster_results, df_train):
    """
    Compara el rendimiento de predicción emocional entre clústeres.
    
    Args:
        cluster_results (dict): Resultados por clúster
        df_train (pd.DataFrame): Datos de entrenamiento completos
    """
    if not cluster_results:
        print("\n❌ No hay resultados para comparar")
        return
    
    print(f"\n📈 COMPARACIÓN DE RENDIMIENTO POR CLÚSTER")
    print("=" * 80)
    
    # Crear DataFrame para análisis
    results_df = pd.DataFrame.from_dict(cluster_results, orient='index')
    results_df = results_df.sort_values('f1_mean', ascending=False)
    
    # === TABLA DE RESULTADOS ===
    print(f"\n🏆 RANKING DE CLÚSTERES POR F1-SCORE:")
    print("-" * 80)
    print(f"{'Rank':4} {'Clúster':8} {'Muestras':9} {'Virales%':9} {'F1-Score':10} {'Precision':10} {'Recall':8} {'ROC-AUC':8}")
    print("-" * 80)
    
    for rank, (idx, row) in enumerate(results_df.iterrows(), 1):
        print(f"{rank:4d} {row['cluster_id']:8d} {row['n_samples']:9d} "
              f"{row['viral_rate']*100:8.1f}% {row['f1_mean']:9.4f} "
              f"{row['precision_mean']:9.4f} {row['recall_mean']:7.4f} "
              f"{row['roc_auc_mean']:7.4f}")
    
    # === ESTADÍSTICAS GENERALES ===
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   🎯 F1-Score promedio: {results_df['f1_mean'].mean():.4f} ± {results_df['f1_mean'].std():.4f}")
    print(f"   🏆 Mejor clúster: {results_df.iloc[0]['cluster_id']} (F1: {results_df.iloc[0]['f1_mean']:.4f})")
    print(f"   🔺 Peor clúster: {results_df.iloc[-1]['cluster_id']} (F1: {results_df.iloc[-1]['f1_mean']:.4f})")
    print(f"   📈 Diferencia máxima: {results_df.iloc[0]['f1_mean'] - results_df.iloc[-1]['f1_mean']:.4f}")
    
    # === ANÁLISIS DE CORRELACIONES ===
    print(f"\n🔍 CORRELACIONES:")
    correlation_viral_rate_f1 = results_df['viral_rate'].corr(results_df['f1_mean'])
    correlation_samples_f1 = results_df['n_samples'].corr(results_df['f1_mean'])
    
    print(f"   📊 Tasa de viralidad vs F1-Score: {correlation_viral_rate_f1:.4f}")
    print(f"   📊 Número de muestras vs F1-Score: {correlation_samples_f1:.4f}")
    
    # === VISUALIZACIÓN ===
    create_cluster_performance_plots(results_df, df_train)


def analyze_cluster_feature_patterns(cluster_results):
    """
    Analiza patrones de features importantes por clúster.
    
    Args:
        cluster_results (dict): Resultados por clúster
    """
    print(f"\n🔍 ANÁLISIS DE PATRONES DE FEATURES POR CLÚSTER")
    print("=" * 60)
    
    # Top 3 clústeres por F1-Score
    sorted_clusters = sorted(cluster_results.items(), 
                           key=lambda x: x[1]['f1_mean'], 
                           reverse=True)
    
    print(f"\n🏆 TOP 3 CLÚSTERES CON MEJOR RENDIMIENTO:")
    
    for rank, (cluster_id, results) in enumerate(sorted_clusters[:3], 1):
        print(f"\n   #{rank} CLÚSTER {cluster_id} (F1: {results['f1_mean']:.4f})")
        print(f"      📊 Muestras: {results['n_samples']}, Virales: {results['viral_rate']*100:.1f}%")
        
        # Top features importantes
        top_features = sorted(results['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"      🎯 Top 5 features importantes:")
        for feat_name, importance in top_features:
            print(f"         {feat_name}: {importance:.4f}")
        
        # Emociones dominantes
        print(f"      😭 Emociones dominantes: {results['top_emotions']}")
        print(f"      😊 Sentimientos dominantes: {results['top_sentiments']}")
    
    print(f"\n❌ BOTTOM 3 CLÚSTERES CON MENOR RENDIMIENTO:")
    
    for rank, (cluster_id, results) in enumerate(sorted_clusters[-3:], 1):
        print(f"\n   #{rank} CLÚSTER {cluster_id} (F1: {results['f1_mean']:.4f})")
        print(f"      📊 Muestras: {results['n_samples']}, Virales: {results['viral_rate']*100:.1f}%")
        
        # Top emociones
        print(f"      😭 Emociones dominantes: {results['top_emotions']}")
        print(f"      😊 Sentimientos dominantes: {results['top_sentiments']}")


def create_cluster_performance_plots(results_df, df_train):
    """
    Crea visualizaciones del rendimiento por clúster.
    
    Args:
        results_df (pd.DataFrame): Resultados de clústeres
        df_train (pd.DataFrame): Datos de entrenamiento
    """
    plt.figure(figsize=(16, 12))
    
    # 1. F1-Score por clúster
    plt.subplot(2, 3, 1)
    bars = plt.bar(results_df['cluster_id'], results_df['f1_mean'], 
                   yerr=results_df['f1_std'], capsize=5)
    plt.xlabel('Clúster ID')
    plt.ylabel('F1-Score')
    plt.title('F1-Score por Clúster')
    plt.xticks(results_df['cluster_id'])
    
    # Colorear barras por rendimiento
    colors = ['red' if x < 0.6 else 'orange' if x < 0.7 else 'green' 
              for x in results_df['f1_mean']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 2. ROC-AUC por clúster
    plt.subplot(2, 3, 2)
    plt.bar(results_df['cluster_id'], results_df['roc_auc_mean'], 
            yerr=results_df['roc_auc_std'], capsize=5, color='skyblue')
    plt.xlabel('Clúster ID')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC por Clúster')
    plt.xticks(results_df['cluster_id'])
    
    # 3. Scatter: Tasa viral vs F1-Score
    plt.subplot(2, 3, 3)
    plt.scatter(results_df['viral_rate'] * 100, results_df['f1_mean'], 
                s=results_df['n_samples']/10, alpha=0.7, c=results_df['cluster_id'], cmap='viridis')
    plt.xlabel('Tasa de Viralidad (%)')
    plt.ylabel('F1-Score')
    plt.title('Tasa Viral vs Rendimiento')
    plt.colorbar(label='Clúster ID')
    
    # Añadir etiquetas de clúster
    for idx, row in results_df.iterrows():
        plt.annotate(f"C{row['cluster_id']}", 
                    (row['viral_rate'] * 100, row['f1_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Distribución de muestras por clúster
    plt.subplot(2, 3, 4)
    plt.bar(results_df['cluster_id'], results_df['n_samples'], color='lightcoral')
    plt.xlabel('Clúster ID')
    plt.ylabel('Número de Muestras')
    plt.title('Distribución de Muestras')
    plt.xticks(results_df['cluster_id'])
    
    # 5. Comparación de métricas (heatmap)
    plt.subplot(2, 3, 5)
    metrics_matrix = results_df[['f1_mean', 'precision_mean', 'recall_mean', 'roc_auc_mean']].T
    sns.heatmap(metrics_matrix, 
                xticklabels=[f'C{int(c)}' for c in results_df['cluster_id']], 
                yticklabels=['F1', 'Precision', 'Recall', 'ROC-AUC'],
                annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
    plt.title('Heatmap de Métricas por Clúster')
    
    # 6. Box plot de F1-Scores
    plt.subplot(2, 3, 6)
    plt.boxplot([results_df['f1_mean']], labels=['F1-Score'])
    plt.scatter([1] * len(results_df), results_df['f1_mean'], 
                c=results_df['cluster_id'], cmap='viridis', alpha=0.7, s=50)
    plt.ylabel('F1-Score')
    plt.title('Distribución de F1-Scores')
    
    # Añadir línea de referencia
    plt.axhline(y=results_df['f1_mean'].mean(), color='red', linestyle='--', 
                label=f'Promedio: {results_df["f1_mean"].mean():.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cluster_emotion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n   💾 Gráfico guardado como 'cluster_emotion_analysis.png'")


if __name__ == '__main__':
    print("🎯 ANÁLISIS DE EMOCIONES POR CLÚSTER K-MEANS")
    print("=" * 80)
    print("🧠 Evaluando si las emociones predicen mejor en ciertos tipos de contenido")
    print("🤖 XGBoost + Validación Cruzada por cada clúster (k=16)")
    print("📊 Métricas: F1, Precision, Recall, ROC-AUC por clúster")
    print("🎭 Identificando patrones emocionales específicos por tipo de contenido")
    print("=" * 80)
    
    analyze_emotions_by_cluster()
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS POR CLÚSTER COMPLETADO")
    print("📊 Resultados disponibles:")
    print("   📈 Tabla de ranking de clústeres por F1-Score")
    print("   🎨 Gráficos: 'cluster_emotion_analysis.png'")
    print("   🔍 Análisis de patterns de features por clúster")
    print("   📋 Correlaciones entre características del clúster y rendimiento")
    print("💡 Conclusiones: Identifica qué tipos de contenido responden mejor a emociones")
    print("🚀 Siguiente paso: Usar mejores clústeres para modelos ensemble")
    print("=" * 80)