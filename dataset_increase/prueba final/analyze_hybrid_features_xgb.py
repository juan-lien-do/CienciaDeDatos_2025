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


def process_four_emotions(df):
    """
    Procesa las 4 emociones específicas desde los campos all_labels y all_scores.
    
    Args:
        df (pd.DataFrame): DataFrame con emotion_all_labels y emotion_all_scores
        
    Returns:
        pd.DataFrame: DataFrame con columnas individuales para cada emoción
    """
    import ast
    
    # Nombres de las 4 emociones que esperamos
    emotions = ['joy', 'anger', 'fear', 'sadness']
    
    # Inicializar columnas para cada emoción
    for emotion in emotions:
        df[f'{emotion}_score'] = 0.0
    
    for idx, row in df.iterrows():
        try:
            # Parsear los strings de labels y scores
            if pd.notna(row['emotion_all_labels']) and pd.notna(row['emotion_all_scores']):
                # Convertir strings a listas
                if isinstance(row['emotion_all_labels'], str):
                    labels = ast.literal_eval(row['emotion_all_labels'])
                else:
                    labels = row['emotion_all_labels']
                
                if isinstance(row['emotion_all_scores'], str):
                    scores = ast.literal_eval(row['emotion_all_scores'])
                else:
                    scores = row['emotion_all_scores']
                
                # Crear diccionario label -> score
                emotion_dict = dict(zip(labels, scores))
                
                # Asignar scores a las columnas correspondientes
                for emotion in emotions:
                    if emotion in emotion_dict:
                        df.loc[idx, f'{emotion}_score'] = emotion_dict[emotion]
                        
        except Exception as e:
            print(f"   ⚠️ Error procesando emociones en registro {idx}: {e}")
            continue
    
    print(f"   ✅ Procesadas 4 emociones: {emotions}")
    return df


def load_hybrid_data_from_neo4j(subset=None):
    """
    Carga datos de Neo4j con features originales + análisis emocional.
    
    Args:
        subset (str, optional): 'train' o 'test' para filtrar datos
        
    Returns:
        pd.DataFrame: Datos con features originales, emocionales y target
    """
    print(f"\n📊 Cargando datos híbridos desde Neo4j...")
    if subset:
        print(f"   🎯 Filtro de subset: {subset}")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Construir query base
            base_query = """
                MATCH (n:Noticia)
                WHERE n.popularity IS NOT NULL
                  AND n.analisis_emocion_titulo_all_labels IS NOT NULL
                  AND n.analisis_emocion_titulo_all_scores IS NOT NULL
            """
            
            # Agregar filtro de subset si se especifica
            if subset:
                base_query += f" AND n.subset = '{subset}'"
            
            # Query completa para obtener features originales + emocionales
            query = base_query + """
                RETURN 
                    n.url AS url,
                    n.titulo AS titulo,
                    n.shares AS shares,
                    n.popularity AS is_viral,
                    n.subset AS subset,
                    
                    // === FEATURES EMOCIONALES (4 EMOCIONES) ===
                    n.analisis_emocion_titulo_all_labels AS emotion_all_labels,
                    n.analisis_emocion_titulo_all_scores AS emotion_all_scores,
                    
                    // === FEATURES ORIGINALES ===
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
                print(f"   ❌ No se encontraron datos con features completas")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Convertir target a entero
            df['is_viral'] = df['is_viral'].astype(int)
            
            # Procesar las 4 emociones específicas
            df = process_four_emotions(df)
            
            print(f"   ✅ Cargados {len(df)} registros híbridos")
            if 'subset' in df.columns:
                print(f"   📈 Distribución por subset:")
                subset_counts = df['subset'].value_counts()
                for subset_name, count in subset_counts.items():
                    viral_count = df[df['subset'] == subset_name]['is_viral'].sum()
                    print(f"      {subset_name}: {count} total, {viral_count} virales")
            
            # Mostrar distribución de las 4 emociones
            if 'joy_score' in df.columns:
                print(f"   🎭 Distribución de emociones por viralidad:")
                for emotion in ['joy', 'anger', 'fear', 'sadness']:
                    col_name = f'{emotion}_score'
                    if col_name in df.columns:
                        viral_mean = df[df['is_viral'] == 1][col_name].mean()
                        non_viral_mean = df[df['is_viral'] == 0][col_name].mean()
                        print(f"      {emotion:8s} - Viral: {viral_mean:.3f}, No Viral: {non_viral_mean:.3f}")
            
            return df
            
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return pd.DataFrame()
        
    finally:
        driver.close()


def analyze_hybrid_feature_statistics(df_train):
    """
    Analiza estadísticas de features originales + emocionales.
    
    Args:
        df_train (pd.DataFrame): Datos de entrenamiento híbridos
    """
    print("\n📊 ANÁLISIS ESTADÍSTICO DE FEATURES HÍBRIDAS")
    print("=" * 70)
    
    # Separar tipos de features
    exclude_cols = ['url', 'titulo', 'shares', 'is_viral', 'subset', 'emotion_all_labels', 'emotion_all_scores']
    emotion_features = ['joy_score', 'anger_score', 'fear_score', 'sadness_score']
    
    # Todas las features disponibles
    all_features = [col for col in df_train.columns if col not in exclude_cols]
    
    # Separar features originales de emocionales
    original_features = [col for col in all_features if col not in emotion_features]
    
    print(f"📋 Resumen de features híbridas:")
    print(f"   🔢 Features originales: {len(original_features)}")
    print(f"   🎭 Features emocionales: {len(emotion_features)}")
    print(f"   📊 Total de features: {len(all_features)}")
    
    # === ANÁLISIS DE FEATURES EMOCIONALES ===
    print(f"\n🎭 ANÁLISIS DE FEATURES EMOCIONALES (4 EMOCIONES):")
    
    # Estadísticas de las 4 emociones por viralidad
    emotions = ['joy', 'anger', 'fear', 'sadness']
    print(f"   📊 Estadísticas de emociones por viralidad:")
    
    for emotion in emotions:
        col_name = f'{emotion}_score'
        if col_name in df_train.columns:
            viral_scores = df_train[df_train['is_viral'] == 1][col_name]
            non_viral_scores = df_train[df_train['is_viral'] == 0][col_name]
            
            print(f"      {emotion.upper():8s}:")
            print(f"         Viral:     μ={viral_scores.mean():.4f}, σ={viral_scores.std():.4f}, max={viral_scores.max():.4f}")
            print(f"         No Viral:  μ={non_viral_scores.mean():.4f}, σ={non_viral_scores.std():.4f}, max={non_viral_scores.max():.4f}")
            
            # Correlación con viralidad
            correlation = df_train[col_name].corr(df_train['is_viral'])
            print(f"         Correlación con viral: {correlation:.4f}")
            print()
    
    # === CORRELACIONES CON TARGET ===
    print(f"\n🎯 CORRELACIÓN CON VIRALIDAD (top 20 features híbridas):")
    
    # Preparar features numéricas para correlación
    numerical_features = df_train.select_dtypes(include=[np.number]).columns
    
    # Calcular correlaciones con is_viral
    correlation_matrix = df_train[list(numerical_features)].corrwith(df_train['is_viral']).abs()
    correlations = correlation_matrix.sort_values(ascending=False)
    
    print(f"   🏆 Top 20 features más correlacionadas:")
    for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
        # Calcular dirección de la correlación (sin valor absoluto)
        direction = "+" if df_train[feature].corr(df_train['is_viral']) > 0 else "-"
        feature_type = "🎭" if feature in emotion_features else "🔢"
        print(f"   {i:2d}. {feature_type} {feature:25s}: {direction}{corr:.4f}")
    
    return all_features, original_features, emotion_features


def prepare_hybrid_features_and_target(df_train, df_test, all_features):
    """
    Prepara features híbridas (originales + emocionales) y target.
    
    Args:
        df_train (pd.DataFrame): Datos de entrenamiento
        df_test (pd.DataFrame): Datos de prueba
        all_features (list): Lista completa de features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler, emotion_encoder)
    """
    print(f"\n🔧 Preparando features híbridas para XGBoost...")
    
    # === PREPARAR FEATURES EMOCIONALES (4 EMOCIONES) ===
    print("   🎭 Usando scores de las 4 emociones directamente...")
    
    # Crear copias para evitar modificar originales
    df_train_processed = df_train.copy()
    df_test_processed = df_test.copy()
    
    # Las 4 emociones ya están procesadas como scores numéricos
    # No necesitamos codificar nada, solo usar directamente
    processed_features = all_features  # Ya incluye joy_score, anger_score, fear_score, sadness_score
    
    # === SEPARAR FEATURES Y TARGET ===
    X_train = df_train_processed[processed_features].copy()
    X_test = df_test_processed[processed_features].copy()
    y_train = df_train_processed['is_viral'].values
    y_test = df_test_processed['is_viral'].values
    
    print(f"   📊 Conjunto de entrenamiento: {X_train.shape}")
    print(f"   📊 Conjunto de prueba: {X_test.shape}")
    
    # === MANEJAR VALORES FALTANTES ===
    print("   🔧 Manejando valores faltantes...")
    
    # Para features numéricas: imputar con mediana
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    # Verificar valores nulos restantes
    null_counts_train = X_train.isnull().sum().sum()
    null_counts_test = X_test.isnull().sum().sum()
    
    if null_counts_train > 0 or null_counts_test > 0:
        print(f"   ⚠️ Valores nulos restantes - Train: {null_counts_train}, Test: {null_counts_test}")
    else:
        print(f"   ✅ No hay valores nulos en los datos")
    
    # === NORMALIZACIÓN SELECTIVA ===
    print("   📏 Normalizando features continuas...")
    
    # Identificar features continuas (no binarias ni categóricas codificadas)
    continuous_features = []
    for col in numerical_cols:
        if X_train[col].nunique() > 10:  # Más conservador para features continuas
            continuous_features.append(col)
    
    scaler = StandardScaler()
    
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
    
    # === ESTADÍSTICAS FINALES ===
    print(f"   🎯 Distribución del target:")
    print(f"      Train - Virales: {y_train.sum()}/{len(y_train)} ({y_train.mean():.3f})")
    print(f"      Test - Virales: {y_test.sum()}/{len(y_test)} ({y_test.mean():.3f})")
    
    print(f"   📊 Features por tipo:")
    emotion_features = ['joy_score', 'anger_score', 'fear_score', 'sadness_score']
    original_count = len([f for f in processed_features if f not in emotion_features])
    emotion_count = len([f for f in processed_features if f in emotion_features])
    print(f"      🔢 Originales: {original_count}")
    print(f"      🎭 Emocionales: {emotion_count}")
    
    # No necesitamos encoders para las 4 emociones (son numéricas)
    encoders = {}
    
    return X_train_scaled, X_test_scaled, y_train, y_test, processed_features, scaler, encoders


def train_hybrid_xgboost_model(X_train, y_train, feature_names):
    """
    Entrena modelo XGBoost híbrido con validación cruzada.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento híbridas
        y_train (np.array): Target de entrenamiento
        feature_names (list): Nombres de las features
        
    Returns:
        dict: Modelo entrenado y métricas de validación cruzada
    """
    print(f"\n🤖 Entrenando modelo XGBoost híbrido...")
    
    # Configurar XGBoost optimizado para features híbridas
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,           # Más árboles para aprovechar features adicionales
        max_depth=7,                # Profundidad mayor para capturar interacciones
        learning_rate=0.08,         # Learning rate ligeramente menor
        gamma=0.2,                  # Regularización moderada
        min_child_weight=3,         # Menos restrictivo para patrones emocionales
        subsample=0.85,             # Submuestreo de filas
        colsample_bytree=0.85,      # Submuestreo de columnas
        reg_alpha=0.02,             # Regularización L1 ligera
        reg_lambda=0.02,            # Regularización L2 ligera
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    print("   ✅ Configuración XGBoost híbrido:")
    print(f"      🌳 Árboles: {xgb_model.n_estimators}")
    print(f"      📊 Profundidad máxima: {xgb_model.max_depth}")
    print(f"      📈 Learning rate: {xgb_model.learning_rate}")
    print(f"      🎭 Features híbridas: {len(feature_names)}")
    
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
    print("   🎯 Entrenando modelo final híbrido...")
    xgb_model.fit(X_train, y_train)
    
    # Mostrar resultados de CV
    print(f"\n📊 RESULTADOS DE VALIDACIÓN CRUZADA HÍBRIDA:")
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


def evaluate_hybrid_model_on_test(model_results, X_test, y_test, feature_names):
    """
    Evalúa el modelo híbrido en el conjunto de prueba.
    
    Args:
        model_results (dict): Resultados del entrenamiento
        X_test (pd.DataFrame): Features de prueba
        y_test (np.array): Target de prueba
        feature_names (list): Nombres de las features
    """
    print(f"\n🧪 EVALUACIÓN HÍBRIDA EN CONJUNTO DE PRUEBA")
    print("=" * 70)
    
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
    
    print(f"📊 MÉTRICAS EN CONJUNTO DE PRUEBA (MODELO HÍBRIDO):")
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
    
    print(f"\n⚖️ COMPARACIÓN CV vs TEST (MODELO HÍBRIDO):")
    print(f"   📊 ROC-AUC - CV: {cv_roc_mean:.4f}, Test: {roc_auc:.4f}, Diff: {abs(cv_roc_mean - roc_auc):.4f}")
    print(f"   📊 F1-Score - CV: {cv_f1_mean:.4f}, Test: {f1:.4f}, Diff: {abs(cv_f1_mean - f1):.4f}")
    
    if abs(cv_roc_mean - roc_auc) < 0.05 and abs(cv_f1_mean - f1) < 0.05:
        print(f"   ✅ Modelo híbrido generaliza bien (diferencias < 0.05)")
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


def analyze_hybrid_feature_importance(model, feature_names, top_n=25):
    """
    Analiza importancia de features híbridas con categorización.
    
    Args:
        model: Modelo XGBoost entrenado
        feature_names (list): Nombres de las features
        top_n (int): Número de top features a mostrar
    """
    print(f"\n🔍 ANÁLISIS DE IMPORTANCIA DE FEATURES HÍBRIDAS (Top {top_n})")
    print("=" * 70)
    
    # Obtener importancias
    importances = model.feature_importances_
    
    # Crear DataFrame para análisis
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Top features
    top_features = feature_importance_df.head(top_n)
    
    # Categorizar features
    def categorize_feature(feature_name):
        if any(emotion in feature_name.lower() for emotion in ['joy_score', 'anger_score', 'fear_score', 'sadness_score']):
            return '🎭 Emociones (4)'
        elif any(word in feature_name.lower() for word in ['channel', 'weekday', 'weekend']):
            return '📅 Canal/Temporal'
        elif any(word in feature_name.lower() for word in ['token', 'word', 'keyword']):
            return '📝 Contenido'
        elif 'lda' in feature_name.lower():
            return '🏷️ Tópicos LDA'
        elif any(word in feature_name.lower() for word in ['polarity', 'subjectivity']):
            return '💭 Sentimiento Global'
        elif any(word in feature_name.lower() for word in ['href', 'img', 'video']):
            return '🖼️ Multimedia'
        else:
            return '🔢 Otras'
    
    print(f"🏆 TOP {top_n} FEATURES HÍBRIDAS MÁS IMPORTANTES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        category = categorize_feature(row['feature'])
        print(f"   {i:2d}. {category} {row['feature']:25s}: {row['importance']:.4f}")
    
    # Análisis por categoría
    print(f"\n📊 IMPORTANCIA POR CATEGORÍA:")
    category_importance = {}
    
    for _, row in feature_importance_df.iterrows():
        category = categorize_feature(row['feature'])
        if category not in category_importance:
            category_importance[category] = 0
        category_importance[category] += row['importance']
    
    # Ordenar categorías por importancia
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    
    total_importance = sum(category_importance.values())
    for category, importance in sorted_categories:
        percentage = (importance / total_importance) * 100
        print(f"   {category}: {importance:.4f} ({percentage:.1f}%)")
    
    # Top features emocionales específicamente (4 emociones)
    emotion_features = feature_importance_df[
        feature_importance_df['feature'].str.contains('joy_score|anger_score|fear_score|sadness_score', case=False, regex=True)
    ]
    
    if len(emotion_features) > 0:
        print(f"\n🎭 IMPORTANCIA DE LAS 4 EMOCIONES:")
        for i, (_, row) in enumerate(emotion_features.iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:15s}: {row['importance']:.4f}")
    
    # Visualización
    create_hybrid_feature_importance_plot(feature_importance_df, category_importance, top_n)
    
    return feature_importance_df


def create_hybrid_feature_importance_plot(feature_importance_df, category_importance, top_n):
    """
    Crea visualizaciones de importancia para el modelo híbrido.
    """
    plt.figure(figsize=(18, 12))
    
    # 1. Top features importantes
    plt.subplot(2, 3, 1)
    top_features = feature_importance_df.head(top_n)
    colors = ['red' if any(emotion in f for emotion in ['joy_score', 'anger_score', 'fear_score', 'sadness_score']) else 'blue' 
              for f in top_features['feature']]
    
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features Híbridas')
    plt.gca().invert_yaxis()
    
    # 2. Importancia por categoría
    plt.subplot(2, 3, 2)
    categories, importances = zip(*sorted(category_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
    plt.pie(importances, labels=categories, autopct='%1.1f%%')
    plt.title('Importancia por Categoría de Feature')
    
    # 3. Distribución de importancias
    plt.subplot(2, 3, 3)
    plt.hist(feature_importance_df['importance'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Importancia')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Importancias')
    plt.axvline(feature_importance_df['importance'].mean(), color='red', linestyle='--',
                label=f'Media: {feature_importance_df["importance"].mean():.4f}')
    plt.legend()
    
    # 4. Comparación emocionales vs originales
    plt.subplot(2, 3, 4)
    emotion_importance = feature_importance_df[
        feature_importance_df['feature'].str.contains('joy_score|anger_score|fear_score|sadness_score', case=False, regex=True)
    ]['importance'].sum()
    
    original_importance = feature_importance_df[
        ~feature_importance_df['feature'].str.contains('joy_score|anger_score|fear_score|sadness_score', case=False, regex=True)
    ]['importance'].sum()
    
    plt.bar(['Features Emocionales', 'Features Originales'], 
            [emotion_importance, original_importance],
            color=['orange', 'lightblue'])
    plt.ylabel('Importancia Total')
    plt.title('Emocionales vs Originales')
    
    # 5. Top 15 features con códigos de color
    plt.subplot(2, 3, 5)
    top_15 = feature_importance_df.head(15)
    
    # Colores por tipo
    color_map = []
    for feature in top_15['feature']:
        if 'joy_score' in feature.lower():
            color_map.append('gold')
        elif 'anger_score' in feature.lower():
            color_map.append('red')
        elif 'fear_score' in feature.lower():
            color_map.append('purple')
        elif 'sadness_score' in feature.lower():
            color_map.append('blue')
        elif any(word in feature.lower() for word in ['lda', 'polarity', 'subjectivity']):
            color_map.append('green')
        else:
            color_map.append('lightblue')
    
    plt.barh(range(len(top_15)), top_15['importance'], color=color_map)
    plt.yticks(range(len(top_15)), [f[:20] + '...' if len(f) > 20 else f 
                                   for f in top_15['feature']])
    plt.xlabel('Importancia')
    plt.title('Top 15 Features (Codificadas por Color)')
    plt.gca().invert_yaxis()
    
    # 6. Acumulado de importancia
    plt.subplot(2, 3, 6)
    cumulative_importance = np.cumsum(top_features['importance'])
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'bo-')
    plt.xlabel('Número de Features')
    plt.ylabel('Importancia Acumulada')
    plt.title('Importancia Acumulada')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_xgboost_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   💾 Gráfico guardado como 'hybrid_xgboost_analysis.png'")


def create_performance_visualization(model_results, test_results):
    """
    Crea visualizaciones del rendimiento del modelo híbrido.
    
    Args:
        model_results (dict): Resultados del entrenamiento
        test_results (dict): Resultados de la evaluación en test
    """
    print(f"\n📊 Creando visualizaciones de rendimiento híbrido...")
    
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
    plt.title('Métricas de Validación Cruzada\n(Modelo Híbrido)')
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
    plt.title('Comparación CV vs Test\n(Modelo Híbrido)')
    plt.legend()
    plt.ylim(0, 1)
    
    # 3. Matriz de confusión
    plt.subplot(2, 3, 3)
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Viral', 'Viral'],
                yticklabels=['No Viral', 'Viral'])
    plt.title('Matriz de Confusión\n(Test Híbrido)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    # 4. ROC-AUC Score
    plt.subplot(2, 3, 4)
    plt.text(0.5, 0.5, f"ROC-AUC: {test_metrics['roc_auc']:.4f}\n\nF1-Score: {test_metrics['f1']:.4f}\n\nAccuracy: {test_metrics['accuracy']:.4f}", 
             ha='center', va='center', fontsize=12,
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.title('Métricas Principales\n(Test Híbrido)')
    plt.axis('off')
    
    # 5. Distribución de probabilidades predichas
    plt.subplot(2, 3, 5)
    probabilities = test_results['probabilities']
    y_test = test_results.get('y_test', [])  # Si está disponible
    
    # Separar por clase predicha
    prob_viral = probabilities[test_results['predictions'] == 1]
    prob_no_viral = probabilities[test_results['predictions'] == 0]
    
    plt.hist(prob_no_viral, bins=20, alpha=0.6, label='Pred: No Viral', color='blue', density=True)
    plt.hist(prob_viral, bins=20, alpha=0.6, label='Pred: Viral', color='red', density=True)
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Densidad')
    plt.title('Distribución de Probabilidades\n(Modelo Híbrido)')
    plt.legend()
    
    # 6. Métricas por fold de CV
    plt.subplot(2, 3, 6)
    fold_data = []
    for i in range(5):  # 5 folds
        fold_scores = [cv_scores[metric][i] for metric in metrics]
        fold_data.append(fold_scores)
    
    fold_data = np.array(fold_data)
    
    for i, metric in enumerate(metrics):
        plt.plot(range(1, 6), fold_data[:, i], 'o-', label=metric.upper().replace('_', '-'), alpha=0.7)
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Métricas por Fold (CV)\n(Modelo Híbrido)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_xgboost_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   💾 Gráfico guardado como 'hybrid_xgboost_performance.png'")


def main():
    """
    Función principal que ejecuta el análisis híbrido completo.
    """
    print("🚀 ANÁLISIS HÍBRIDO: FEATURES ORIGINALES + EMOCIONALES")
    print("=" * 80)
    print("🎯 Objetivo: Predecir viralidad combinando features originales + análisis emocional")
    print("🤖 Modelo: XGBoost híbrido con validación en conjunto independiente")
    print("📊 Pipeline: Features Originales + Emociones → XGBoost → Evaluación")
    print("🎭 Novedad: Incorpora análisis de emociones y sentimientos del título")
    print("=" * 80)
    
    # 1. Cargar datos híbridos de entrenamiento y prueba
    print("📊 Cargando conjuntos de datos híbridos...")
    df_train = load_hybrid_data_from_neo4j(subset='train')
    df_test = load_hybrid_data_from_neo4j(subset='test')
    
    if df_train.empty or df_test.empty:
        print("❌ No se pudieron cargar los datos híbridos necesarios")
        return
    
    # 2. Análisis exploratorio de features híbridas
    all_features, original_features, emotion_features = analyze_hybrid_feature_statistics(df_train)
    
    # 3. Preparar datos híbridos
    X_train, X_test, y_train, y_test, feature_names, scaler, encoders = prepare_hybrid_features_and_target(
        df_train, df_test, all_features
    )
    
    # 4. Entrenar modelo XGBoost híbrido
    model_results = train_hybrid_xgboost_model(X_train, y_train, feature_names)
    
    # 5. Evaluar en conjunto de prueba
    test_results = evaluate_hybrid_model_on_test(model_results, X_test, y_test, feature_names)
    
    # 6. Análisis de importancia de features híbridas
    feature_importance_df = analyze_hybrid_feature_importance(
        model_results['model'], feature_names, top_n=25
    )
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS HÍBRIDO COMPLETADO")
    print("📊 Resumen de resultados:")
    print(f"   🎯 ROC-AUC (CV): {model_results['cv_scores']['roc_auc'].mean():.4f}")
    print(f"   🎯 ROC-AUC (Test): {test_results['test_metrics']['roc_auc']:.4f}")
    print(f"   🎯 F1-Score (Test): {test_results['test_metrics']['f1']:.4f}")
    print(f"   🔢 Features originales: {len(original_features)}")
    print(f"   🎭 Features emocionales: {len(emotion_features)}")
    print(f"   📊 Total features: {len(feature_names)}")
    print("📁 Archivos generados:")
    print("   🎨 hybrid_xgboost_analysis.png")
    print("💡 El modelo híbrido combina lo mejor de ambos mundos")
    print("🚀 Compara resultados con modelo solo-features-originales")
    print("=" * 80)


if __name__ == '__main__':
    main()