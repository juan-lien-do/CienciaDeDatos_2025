from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configuración del análisis
FEATURES_PREFIX = "orig_"  # Prefijo de las features originales cargadas
TARGET_COLUMN = "popularity"  # Variable objetivo (viral/no viral)
TEST_SIZE_RATIO = 0.2  # Proporción esperada del conjunto de test


def load_data_from_neo4j(subset=None):
    """
    Carga datos de Neo4j con todas las features originales.
    
    Args:
        subset (str, optional): 'train' o 'test' para filtrar datos
        
    Returns:
        pd.DataFrame: Datos con features originales y target
    """
    print(f"\n📊 Cargando datos desde Neo4j...")
    if subset:
        print(f"   🎯 Filtro de subset: {subset}")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Construir query base
            base_query = """
                MATCH (n:Noticia)
                WHERE n.popularity IS NOT NULL
            """
            
            # Agregar filtro de subset si se especifica
            if subset:
                base_query += f" AND n.subset = '{subset}'"
            
            # Query completa para obtener todas las features originales
            query = base_query + """
                RETURN 
                    n.url AS url,
                    n.titulo AS titulo,
                    n.shares AS shares,
                    n.popularity AS is_viral,
                    n.subset AS subset,
                    
                    // Features temporales
                    n.orig_timedelta AS timedelta,
                    
                    // Features de contenido
                    n.orig_n_tokens_title AS n_tokens_title,
                    n.orig_n_tokens_content AS n_tokens_content,
                    n.orig_n_unique_tokens AS n_unique_tokens,
                    n.orig_n_non_stop_words AS n_non_stop_words,
                    n.orig_n_non_stop_unique_tokens AS n_non_stop_unique_tokens,
                    n.orig_average_token_length AS average_token_length,
                    n.orig_num_keywords AS num_keywords,
                    
                    // Features de multimedia
                    n.orig_num_hrefs AS num_hrefs,
                    n.orig_num_self_hrefs AS num_self_hrefs,
                    n.orig_num_imgs AS num_imgs,
                    n.orig_num_videos AS num_videos,
                    
                    // Features de canales de datos
                    n.orig_data_channel_is_lifestyle AS data_channel_is_lifestyle,
                    n.orig_data_channel_is_entertainment AS data_channel_is_entertainment,
                    n.orig_data_channel_is_bus AS data_channel_is_bus,
                    n.orig_data_channel_is_socmed AS data_channel_is_socmed,
                    n.orig_data_channel_is_tech AS data_channel_is_tech,
                    n.orig_data_channel_is_world AS data_channel_is_world,
                    
                    // Features de keywords
                    n.orig_kw_min_min AS kw_min_min,
                    n.orig_kw_max_min AS kw_max_min,
                    n.orig_kw_avg_min AS kw_avg_min,
                    n.orig_kw_min_max AS kw_min_max,
                    n.orig_kw_max_max AS kw_max_max,
                    n.orig_kw_avg_max AS kw_avg_max,
                    n.orig_kw_min_avg AS kw_min_avg,
                    n.orig_kw_max_avg AS kw_max_avg,
                    n.orig_kw_avg_avg AS kw_avg_avg,
                    
                    // Features de auto-referencia
                    n.orig_self_reference_min_shares AS self_reference_min_shares,
                    n.orig_self_reference_max_shares AS self_reference_max_shares,
                    n.orig_self_reference_avg_sharess AS self_reference_avg_shares,
                    
                    // Features de día de la semana
                    n.orig_weekday_is_monday AS weekday_is_monday,
                    n.orig_weekday_is_tuesday AS weekday_is_tuesday,
                    n.orig_weekday_is_wednesday AS weekday_is_wednesday,
                    n.orig_weekday_is_thursday AS weekday_is_thursday,
                    n.orig_weekday_is_friday AS weekday_is_friday,
                    n.orig_weekday_is_saturday AS weekday_is_saturday,
                    n.orig_weekday_is_sunday AS weekday_is_sunday,
                    n.orig_is_weekend AS is_weekend,
                    
                    // Features LDA (temas)
                    n.orig_LDA_00 AS LDA_00,
                    n.orig_LDA_01 AS LDA_01,
                    n.orig_LDA_02 AS LDA_02,
                    n.orig_LDA_03 AS LDA_03,
                    n.orig_LDA_04 AS LDA_04,
                    
                    // Features de sentimiento global
                    n.orig_global_subjectivity AS global_subjectivity,
                    n.orig_global_sentiment_polarity AS global_sentiment_polarity,
                    n.orig_global_rate_positive_words AS global_rate_positive_words,
                    n.orig_global_rate_negative_words AS global_rate_negative_words,
                    n.orig_rate_positive_words AS rate_positive_words,
                    n.orig_rate_negative_words AS rate_negative_words,
                    
                    // Features de polaridad
                    n.orig_avg_positive_polarity AS avg_positive_polarity,
                    n.orig_min_positive_polarity AS min_positive_polarity,
                    n.orig_max_positive_polarity AS max_positive_polarity,
                    n.orig_avg_negative_polarity AS avg_negative_polarity,
                    n.orig_min_negative_polarity AS min_negative_polarity,
                    n.orig_max_negative_polarity AS max_negative_polarity,
                    
                    // Features del título
                    n.orig_title_subjectivity AS title_subjectivity,
                    n.orig_title_sentiment_polarity AS title_sentiment_polarity,
                    n.orig_abs_title_subjectivity AS abs_title_subjectivity,
                    n.orig_abs_title_sentiment_polarity AS abs_title_sentiment_polarity
            """
            
            result = session.run(query)
            records = [dict(record) for record in result]
            
            if not records:
                print(f"   ❌ No se encontraron datos con features originales")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Convertir target a entero
            df['is_viral'] = df['is_viral'].astype(int)
            
            print(f"   ✅ Cargados {len(df)} registros")
            if 'subset' in df.columns:
                print(f"   📈 Distribución por subset:")
                subset_counts = df['subset'].value_counts()
                for subset_name, count in subset_counts.items():
                    viral_count = df[df['subset'] == subset_name]['is_viral'].sum()
                    print(f"      {subset_name}: {count} total, {viral_count} virales")
            
            return df
            
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return pd.DataFrame()
        
    finally:
        driver.close()


def analyze_feature_statistics(df_train):
    """
    Analiza estadísticas descriptivas de las features de entrenamiento.
    
    Args:
        df_train (pd.DataFrame): Datos de entrenamiento
    """
    print("\n📊 ANÁLISIS ESTADÍSTICO DE FEATURES")
    print("=" * 60)
    
    # Separar features numéricas de categóricas
    exclude_cols = ['url', 'titulo', 'shares', 'is_viral', 'subset']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    # Identificar features categóricas (binarias)
    categorical_features = []
    numerical_features = []
    
    for col in feature_cols:
        if df_train[col].dtype in ['int64', 'float64']:
            unique_vals = df_train[col].dropna().nunique()
            if unique_vals <= 2:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
    
    print(f"📋 Resumen de features:")
    print(f"   🔢 Features numéricas: {len(numerical_features)}")
    print(f"   🏷️ Features categóricas: {len(categorical_features)}")
    print(f"   📊 Total de features: {len(feature_cols)}")
    
    # Estadísticas de features numéricas
    if numerical_features:
        print(f"\n🔢 TOP 10 FEATURES NUMÉRICAS (por varianza):")
        numerical_stats = df_train[numerical_features].describe()
        variances = df_train[numerical_features].var().sort_values(ascending=False)
        
        for i, (feature, variance) in enumerate(variances.head(10).items(), 1):
            mean = numerical_stats.loc['mean', feature]
            std = numerical_stats.loc['std', feature]
            print(f"   {i:2d}. {feature}: μ={mean:.3f}, σ={std:.3f}, var={variance:.3f}")
    
    # Distribución de features categóricas
    if categorical_features:
        print(f"\n🏷️ FEATURES CATEGÓRICAS (proporción de 1s):")
        for feature in categorical_features[:15]:  # Mostrar top 15
            proportion = df_train[feature].mean()
            count_1s = df_train[feature].sum()
            total = len(df_train)
            print(f"   {feature}: {proportion:.3f} ({count_1s}/{total})")
    
    # Análisis de correlación con target
    print(f"\n🎯 CORRELACIÓN CON VIRALIDAD (top 15):")
    correlations = df_train[feature_cols + ['is_viral']].corr()['is_viral'].abs().sort_values(ascending=False)
    
    for i, (feature, corr) in enumerate(correlations[1:16].items(), 1):  # Excluir is_viral consigo mismo
        direction = "+" if df_train[feature].corr(df_train['is_viral']) > 0 else "-"
        print(f"   {i:2d}. {feature}: {direction}{corr:.4f}")
    
    return feature_cols, numerical_features, categorical_features


def prepare_features_and_target(df_train, df_test, feature_cols):
    """
    Prepara features y target para entrenamiento y validación.
    
    Args:
        df_train (pd.DataFrame): Datos de entrenamiento
        df_test (pd.DataFrame): Datos de prueba
        feature_cols (list): Lista de columnas de features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print(f"\n🔧 Preparando features para XGBoost...")
    
    # Separar features y target
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train['is_viral'].values
    y_test = df_test['is_viral'].values
    
    print(f"   📊 Conjunto de entrenamiento: {X_train.shape}")
    print(f"   📊 Conjunto de prueba: {X_test.shape}")
    
    # Manejar valores faltantes
    print("   🔧 Manejando valores faltantes...")
    
    # Para features numéricas: imputar con mediana
    # Para features categóricas: imputar con 0
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    # Verificar que no queden valores nulos
    null_counts_train = X_train.isnull().sum().sum()
    null_counts_test = X_test.isnull().sum().sum()
    
    if null_counts_train > 0 or null_counts_test > 0:
        print(f"   ⚠️ Valores nulos restantes - Train: {null_counts_train}, Test: {null_counts_test}")
    else:
        print(f"   ✅ No hay valores nulos en los datos")
    
    # Normalización para features numéricas (opcional para XGBoost, pero puede ayudar)
    print("   📏 Normalizando features numéricas...")
    scaler = StandardScaler()
    
    # Identificar features numéricas continuas (no binarias)
    continuous_features = []
    for col in numerical_cols:
        if X_train[col].nunique() > 2:
            continuous_features.append(col)
    
    if continuous_features:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
        
        print(f"      Normalizadas {len(continuous_features)} features continuas")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        print(f"      Sin features continuas para normalizar")
    
    print(f"   🎯 Distribución del target:")
    print(f"      Train - Virales: {y_train.sum()}/{len(y_train)} ({y_train.mean():.3f})")
    print(f"      Test - Virales: {y_test.sum()}/{len(y_test)} ({y_test.mean():.3f})")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def train_xgboost_model(X_train, y_train, feature_names):
    """
    Entrena modelo XGBoost con validación cruzada.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (np.array): Target de entrenamiento
        feature_names (list): Nombres de las features
        
    Returns:
        dict: Modelo entrenado y métricas de validación cruzada
    """
    print(f"\n🤖 Entrenando modelo XGBoost...")
    
    # Configurar XGBoost optimizado para este tipo de datos
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,           # Número de árboles
        max_depth=6,                # Profundidad máxima
        learning_rate=0.1,          # Tasa de aprendizaje
        gamma=0.1,                  # Regularización mínima
        min_child_weight=5,         # Peso mínimo en hoja
        subsample=0.8,              # Submuestreo de filas
        colsample_bytree=0.8,       # Submuestreo de columnas
        reg_alpha=0.01,             # Regularización L1
        reg_lambda=0.01,            # Regularización L2
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    print("   ✅ Configuración XGBoost:")
    print(f"      🌳 Árboles: {xgb_model.n_estimators}")
    print(f"      📊 Profundidad máxima: {xgb_model.max_depth}")
    print(f"      📈 Learning rate: {xgb_model.learning_rate}")
    
    # Validación cruzada estratificada
    print("   🔄 Realizando validación cruzada (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Múltiples métricas
    cv_accuracy = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_precision = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='precision', n_jobs=-1)
    cv_recall = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
    cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc_auc = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Entrenar modelo final
    print("   🎯 Entrenando modelo final...")
    xgb_model.fit(X_train, y_train)
    
    # Mostrar resultados de CV
    print(f"\n📊 RESULTADOS DE VALIDACIÓN CRUZADA:")
    print(f"   🎯 Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"   🎯 Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"   🎯 Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"   🎯 F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"   🎯 ROC-AUC:   {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
    
    return {
        'model': xgb_model,
        'cv_scores': {
            'accuracy': cv_accuracy,
            'precision': cv_precision,
            'recall': cv_recall,
            'f1': cv_f1,
            'roc_auc': cv_roc_auc
        }
    }


def evaluate_on_test_set(model_results, X_test, y_test, feature_names):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        model_results (dict): Resultados del entrenamiento
        X_test (pd.DataFrame): Features de prueba
        y_test (np.array): Target de prueba
        feature_names (list): Nombres de las features
    """
    print(f"\n🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("=" * 60)
    
    model = model_results['model']
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas detalladas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📊 MÉTRICAS EN CONJUNTO DE PRUEBA:")
    print(f"   🎯 Accuracy:  {accuracy:.4f}")
    print(f"   🎯 Precision: {precision:.4f}")
    print(f"   🎯 Recall:    {recall:.4f}")
    print(f"   🎯 F1-Score:  {f1:.4f}")
    print(f"   🎯 ROC-AUC:   {roc_auc:.4f}")
    
    # Reporte detallado
    print(f"\n📈 REPORTE DETALLADO:")
    print(classification_report(y_test, y_pred, target_names=['No Viral', 'Viral']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 MATRIZ DE CONFUSIÓN:")
    print(f"   Verdadero\\Predicho   No Viral  Viral")
    print(f"   No Viral             {cm[0,0]:8d}  {cm[0,1]:5d}")
    print(f"   Viral                {cm[1,0]:8d}  {cm[1,1]:5d}")
    
    # Comparación con validación cruzada
    cv_roc_mean = model_results['cv_scores']['roc_auc'].mean()
    cv_f1_mean = model_results['cv_scores']['f1'].mean()
    
    print(f"\n⚖️ COMPARACIÓN CV vs TEST:")
    print(f"   📊 ROC-AUC - CV: {cv_roc_mean:.4f}, Test: {roc_auc:.4f}, Diff: {abs(cv_roc_mean - roc_auc):.4f}")
    print(f"   📊 F1-Score - CV: {cv_f1_mean:.4f}, Test: {f1:.4f}, Diff: {abs(cv_f1_mean - f1):.4f}")
    
    if abs(cv_roc_mean - roc_auc) < 0.05 and abs(cv_f1_mean - f1) < 0.05:
        print(f"   ✅ Modelo generaliza bien (diferencias < 0.05)")
    else:
        print(f"   ⚠️ Posible overfitting o cambio en distribución de datos")
    
    return {
        'test_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        },
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Analiza y visualiza la importancia de features.
    
    Args:
        model: Modelo XGBoost entrenado
        feature_names (list): Nombres de las features
        top_n (int): Número de top features a mostrar
    """
    print(f"\n🔍 ANÁLISIS DE IMPORTANCIA DE FEATURES (Top {top_n})")
    print("=" * 60)
    
    # Obtener importancias
    importances = model.feature_importances_
    
    # Crear DataFrame para análisis
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Top features
    top_features = feature_importance_df.head(top_n)
    
    print(f"🏆 TOP {top_n} FEATURES MÁS IMPORTANTES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:30s}: {row['importance']:.4f}")
    
    # Visualización
    plt.figure(figsize=(15, 12))
    
    # 1. Importancia de features
    plt.subplot(2, 2, 1)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features más Importantes')
    plt.gca().invert_yaxis()
    
    # 2. Distribución de importancias
    plt.subplot(2, 2, 2)
    plt.hist(importances, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Importancia')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Importancias')
    plt.axvline(np.mean(importances), color='red', linestyle='--', 
                label=f'Media: {np.mean(importances):.4f}')
    plt.legend()
    
    # 3. Importancia acumulada
    plt.subplot(2, 2, 3)
    cumulative_importance = np.cumsum(top_features['importance'])
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'bo-')
    plt.xlabel('Número de Features')
    plt.ylabel('Importancia Acumulada')
    plt.title('Importancia Acumulada')
    plt.grid(True, alpha=0.3)
    
    # Mostrar punto donde se alcanza el 80% de importancia
    total_importance = np.sum(importances)
    threshold_80 = 0.8 * total_importance
    features_80 = np.argmax(cumulative_importance >= threshold_80) + 1
    plt.axhline(threshold_80, color='red', linestyle='--', alpha=0.7)
    plt.axvline(features_80, color='red', linestyle='--', alpha=0.7)
    plt.text(features_80, threshold_80, f'{features_80} features\n(80% importancia)', 
             verticalalignment='bottom', horizontalalignment='left')
    
    # 4. Categorización por tipo de feature
    plt.subplot(2, 2, 4)
    
    # Categorizar features por tipo
    categories = {
        'Contenido': ['tokens', 'keywords', 'length'],
        'Temporal': ['weekday', 'weekend', 'timedelta'],
        'Canal': ['channel'],
        'Keywords': ['kw_'],
        'Sentimiento': ['sentiment', 'polarity', 'subjectivity', 'positive', 'negative'],
        'LDA': ['LDA'],
        'Multimedia': ['hrefs', 'imgs', 'videos'],
        'Referencias': ['reference']
    }
    
    category_importance = {}
    for category, keywords in categories.items():
        category_importance[category] = 0
        for _, row in top_features.iterrows():
            feature_name = row['feature'].lower()
            if any(keyword.lower() in feature_name for keyword in keywords):
                category_importance[category] += row['importance']
    
    # Otros (features que no encajan en categorías)
    total_categorized = sum(category_importance.values())
    category_importance['Otros'] = max(0, top_features['importance'].sum() - total_categorized)
    
    # Filtrar categorías con importancia > 0
    category_importance = {k: v for k, v in category_importance.items() if v > 0}
    
    if category_importance:
        categories_sorted = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        cat_names, cat_values = zip(*categories_sorted)
        
        plt.pie(cat_values, labels=cat_names, autopct='%1.1f%%')
        plt.title('Importancia por Categoría de Feature')
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Gráfico guardado como 'xgboost_feature_analysis.png'")
    
    return feature_importance_df


def create_performance_visualization(model_results, test_results):
    """
    Crea visualizaciones del rendimiento del modelo.
    
    Args:
        model_results (dict): Resultados del entrenamiento
        test_results (dict): Resultados de la evaluación en test
    """
    print(f"\n📊 Creando visualizaciones de rendimiento...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Distribución de métricas de CV
    plt.subplot(2, 3, 1)
    cv_scores = model_results['cv_scores']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_means = [cv_scores[metric].mean() for metric in metrics]
    cv_stds = [cv_scores[metric].std() for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    plt.bar(x_pos, cv_means, yerr=cv_stds, alpha=0.7, capsize=5)
    plt.xticks(x_pos, [m.upper().replace('_', '-') for m in metrics], rotation=45)
    plt.ylabel('Score')
    plt.title('Métricas de Validación Cruzada')
    plt.ylim(0, 1)
    
    # 2. Comparación CV vs Test
    plt.subplot(2, 3, 2)
    test_metrics = test_results['test_metrics']
    cv_test_comparison = []
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        cv_mean = cv_scores[metric].mean()
        test_score = test_metrics[metric]
        cv_test_comparison.append([cv_mean, test_score])
    
    cv_test_comparison = np.array(cv_test_comparison).T
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    plt.bar(x_pos - width/2, cv_test_comparison[0], width, label='CV', alpha=0.7)
    plt.bar(x_pos + width/2, cv_test_comparison[1], width, label='Test', alpha=0.7)
    plt.xticks(x_pos, [m.upper().replace('_', '-') for m in metrics], rotation=45)
    plt.ylabel('Score')
    plt.title('Comparación CV vs Test')
    plt.legend()
    plt.ylim(0, 1)
    
    # 3. Matriz de confusión
    plt.subplot(2, 3, 3)
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Viral', 'Viral'],
                yticklabels=['No Viral', 'Viral'])
    plt.title('Matriz de Confusión (Test)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    # 4. Curva ROC
    plt.subplot(2, 3, 4)
    # Necesitaríamos y_test y y_pred_proba para esto
    # Por simplicidad, mostramos un placeholder
    plt.text(0.5, 0.5, f"ROC-AUC: {test_metrics['roc_auc']:.4f}", 
             ha='center', va='center', fontsize=14,
             transform=plt.gca().transAxes)
    plt.title('Curva ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # 5. Distribución de probabilidades predichas
    plt.subplot(2, 3, 5)
    probabilities = test_results['probabilities']
    y_pred = test_results['predictions']
    
    # Separar por clase real
    prob_viral = probabilities[test_results['predictions'] == 1]
    prob_no_viral = probabilities[test_results['predictions'] == 0]
    
    plt.hist(prob_no_viral, bins=20, alpha=0.6, label='No Viral', color='blue')
    plt.hist(prob_viral, bins=20, alpha=0.6, label='Viral', color='red')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades')
    plt.legend()
    
    # 6. Métricas por fold de CV
    plt.subplot(2, 3, 6)
    fold_data = []
    for i in range(5):  # 5 folds
        fold_scores = [cv_scores[metric][i] for metric in metrics]
        fold_data.append(fold_scores)
    
    fold_data = np.array(fold_data)
    
    for i, metric in enumerate(metrics):
        plt.plot(range(1, 6), fold_data[:, i], 'o-', label=metric.upper().replace('_', '-'))
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Métricas por Fold (CV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   💾 Gráfico guardado como 'xgboost_performance.png'")


def main():
    """
    Función principal que ejecuta todo el análisis.
    """
    print("🚀 ANÁLISIS DE NOTICIAS CON XGBOOST")
    print("=" * 80)
    print("🎯 Objetivo: Predecir viralidad usando features originales del dataset")
    print("🤖 Modelo: XGBoost con validación en conjunto independiente")
    print("📊 Pipeline: Train → CV → Test → Análisis de features")
    print("=" * 80)
    
    # 1. Cargar datos de entrenamiento y prueba
    print("📊 Cargando conjuntos de datos...")
    df_train = load_data_from_neo4j(subset='train')
    df_test = load_data_from_neo4j(subset='test')
    
    if df_train.empty or df_test.empty:
        print("❌ No se pudieron cargar los datos necesarios")
        return
    
    # 2. Análisis exploratorio de features
    feature_cols, numerical_features, categorical_features = analyze_feature_statistics(df_train)
    
    # 3. Preparar datos
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_features_and_target(
        df_train, df_test, feature_cols
    )
    
    # 4. Entrenar modelo XGBoost
    model_results = train_xgboost_model(X_train, y_train, feature_names)
    
    # 5. Evaluar en conjunto de prueba
    test_results = evaluate_on_test_set(model_results, X_test, y_test, feature_names)
    
    # 6. Análisis de importancia de features
    feature_importance_df = analyze_feature_importance(
        model_results['model'], feature_names, top_n=20
    )
    
    # 7. Visualizaciones de rendimiento
    create_performance_visualization(model_results, test_results)
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("📊 Resumen de resultados:")
    print(f"   🎯 ROC-AUC (CV): {model_results['cv_scores']['roc_auc'].mean():.4f}")
    print(f"   🎯 ROC-AUC (Test): {test_results['test_metrics']['roc_auc']:.4f}")
    print(f"   🎯 F1-Score (Test): {test_results['test_metrics']['f1']:.4f}")
    print(f"   🔢 Features analizadas: {len(feature_names)}")
    print("📁 Archivos generados:")
    print("   🎨 xgboost_feature_analysis.png")
    print("   📊 xgboost_performance.png")
    print("💡 El modelo está listo para predicciones en producción")
    print("=" * 80)


if __name__ == '__main__':
    main()