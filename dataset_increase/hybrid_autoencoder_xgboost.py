from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configurar TensorFlow para usar GPU si está disponible
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0] if tf.config.experimental.list_physical_devices('GPU') else None, True) if tf.config.experimental.list_physical_devices('GPU') else None


def build_autoencoder():
    """
    Construye un autoencoder para reducir embeddings de 384D a 64D.
    
    Arquitectura:
    - Entrada: 384 dimensiones
    - Encoder: Dense(128, relu) -> Dropout(0.2) -> Dense(64, relu) [Capa Latente]
    - Decoder: Dense(128, relu) -> Dense(384, linear)
    
    Returns:
        tuple: (autoencoder_completo, encoder_solo)
    """
    print("🧠 Construyendo arquitectura del Autoencoder...")
    
    # === ENCODER ===
    input_layer = layers.Input(shape=(384,), name='input_embeddings')
    
    # Primera capa del encoder
    encoded = layers.Dense(128, activation='relu', name='encoder_dense_1')(input_layer)
    encoded = layers.Dropout(0.2, name='encoder_dropout')(encoded)
    
    # Capa latente (bottleneck)
    latent = layers.Dense(64, activation='relu', name='latent_space')(encoded)
    
    # === DECODER ===
    decoded = layers.Dense(128, activation='relu', name='decoder_dense_1')(latent)
    output_layer = layers.Dense(384, activation='linear', name='reconstructed_embeddings')(decoded)
    
    # === MODELOS ===
    # Autoencoder completo (entrada -> reconstrucción)
    autoencoder = keras.Model(input_layer, output_layer, name='autoencoder')
    
    # Encoder solo (entrada -> representación latente)
    encoder = keras.Model(input_layer, latent, name='encoder')
    
    # Compilar autoencoder
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("   ✅ Autoencoder construido:")
    print(f"      📥 Entrada: 384 dimensiones (embeddings)")
    print(f"      🔄 Capa latente: 64 dimensiones")
    print(f"      📤 Salida: 384 dimensiones (reconstrucción)")
    print(f"      🎯 Optimizador: Adam | Loss: MSE")
    
    return autoencoder, encoder


def load_training_data_with_embeddings():
    """
    Carga datos de entrenamiento incluyendo embeddings limpios y análisis emocional.
    
    Returns:
        pd.DataFrame: Datos completos de entrenamiento
    """
    print("\n📊 Cargando datos con embeddings y emociones...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            query = """
                MATCH (n:Noticia)
                WHERE n.subset = 'train'
                  AND n.popularity IS NOT NULL
                  AND n.embedding_titulo_clean IS NOT NULL
                  AND n.analisis_sentimiento_titulo_label IS NOT NULL
                  AND n.analisis_emocion_titulo_label IS NOT NULL
                RETURN 
                    n.titulo AS titulo,
                    n.popularity AS is_viral,
                    n.embedding_titulo_clean AS embeddings,
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
                print("   ❌ No se encontraron datos con embeddings limpios y emociones")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            print(f"   ✅ Cargados {len(df)} registros completos")
            print(f"   📈 Distribución de viralidad: {df['is_viral'].value_counts().to_dict()}")
            
            return df
            
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return pd.DataFrame()
        
    finally:
        driver.close()


def prepare_features_for_hybrid_model(df):
    """
    Prepara embeddings y features emocionales para el modelo híbrido.
    
    Args:
        df (pd.DataFrame): Datos de entrenamiento
        
    Returns:
        tuple: (X_embeddings, X_emotion_features, y, emotion_feature_names)
    """
    print("\n🔧 Preparando features para modelo híbrido...")
    
    # === EMBEDDINGS (384D) ===
    print("   📊 Extrayendo embeddings de 384D...")
    embeddings_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        if isinstance(row['embeddings'], list) and len(row['embeddings']) == 384:
            embeddings_list.append(row['embeddings'])
            valid_indices.append(idx)
    
    if not embeddings_list:
        raise ValueError("No se encontraron embeddings válidos de 384D")
    
    X_embeddings = np.array(embeddings_list)
    
    # Filtrar DataFrame solo con índices válidos
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"      ✅ Embeddings: {X_embeddings.shape}")
    
    # === FEATURES EMOCIONALES ===
    print("   🎭 Preparando features emocionales...")
    emotion_df = pd.DataFrame()
    
    # One-hot encoding para emociones y sentimientos principales
    emotion_dummies = pd.get_dummies(df_valid['emotion_label'], prefix='emotion')
    sentiment_dummies = pd.get_dummies(df_valid['sentiment_label'], prefix='sentiment')
    
    emotion_df = pd.concat([emotion_df, emotion_dummies, sentiment_dummies], axis=1)
    
    # Scores de confianza
    emotion_df['emotion_confidence'] = df_valid['emotion_score'].values
    emotion_df['sentiment_confidence'] = df_valid['sentiment_score'].values
    
    # Features específicas por emoción y sentimiento
    for idx, row in df_valid.iterrows():
        # Emociones específicas
        if isinstance(row['emotion_all_labels'], list) and isinstance(row['emotion_all_scores'], list):
            emotion_dict = dict(zip(row['emotion_all_labels'], row['emotion_all_scores']))
            
            for emotion in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'optimism']:
                score = emotion_dict.get(emotion, 0.0)
                col_name = f'emotion_score_{emotion}'
                if col_name not in emotion_df.columns:
                    emotion_df[col_name] = 0.0
                emotion_df.at[idx, col_name] = score
        
        # Sentimientos específicos
        if isinstance(row['sentiment_all_labels'], list) and isinstance(row['sentiment_all_scores'], list):
            sentiment_dict = dict(zip(row['sentiment_all_labels'], row['sentiment_all_scores']))
            
            for sentiment in ['negative', 'neutral', 'positive']:
                score = sentiment_dict.get(sentiment, 0.0)
                col_name = f'sentiment_score_{sentiment}'
                if col_name not in emotion_df.columns:
                    emotion_df[col_name] = 0.0
                emotion_df.at[idx, col_name] = score
    
    # Rellenar valores faltantes
    emotion_df = emotion_df.fillna(0)
    X_emotion_features = emotion_df.values
    
    # Target
    y = df_valid['is_viral'].astype(int).values
    
    print(f"      ✅ Features emocionales: {X_emotion_features.shape}")
    print(f"      🎯 Target: {y.sum()} virales / {len(y)} total")
    
    return X_embeddings, X_emotion_features, y, emotion_df.columns.tolist()


def clean_emotion_features_iqr(X_emotion):
    """
    Aplica Winsorization basada en IQR a las features emocionales.
    
    Args:
        X_emotion (np.array): Matriz de features emocionales
        
    Returns:
        np.array: Features emocionales con outliers truncados
    """
    print("   📐 Calculando límites IQR para cada feature emocional...")
    
    X_emotion_df = pd.DataFrame(X_emotion)
    X_emotion_clean = X_emotion_df.copy()
    
    outliers_count = 0
    
    for col in X_emotion_df.columns:
        # Calcular Q1, Q3 e IQR
        Q1 = X_emotion_df[col].quantile(0.25)
        Q3 = X_emotion_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Límites de Winsorization (1.5 * IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contar outliers antes de truncar
        outliers_before = ((X_emotion_df[col] < lower_bound) | (X_emotion_df[col] > upper_bound)).sum()
        outliers_count += outliers_before
        
        # Aplicar Winsorization (truncamiento)
        X_emotion_clean[col] = X_emotion_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"      ✂️ Features emocionales truncadas: {outliers_count} valores atípicos")
    
    return X_emotion_clean.values


def detect_embedding_outliers(X_embed, contamination=0.01):
    """
    Detecta outliers en embeddings usando Isolation Forest.
    
    Args:
        X_embed (np.array): Matriz de embeddings (384D)
        contamination (float): Proporción esperada de outliers
        
    Returns:
        np.array: Máscara booleana (True = inlier, False = outlier)
    """
    print(f"   🌲 Configurando Isolation Forest (contamination={contamination})...")
    
    # Configurar Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Ajustar y predecir outliers
    outlier_predictions = iso_forest.fit_predict(X_embed)
    
    # Convertir a máscara booleana (1 = inlier, -1 = outlier)
    inlier_mask = outlier_predictions == 1
    
    outliers_detected = (~inlier_mask).sum()
    outlier_percentage = (outliers_detected / len(X_embed)) * 100
    
    print(f"      🔍 Outliers detectados: {outliers_detected} ({outlier_percentage:.2f}%)")
    print(f"      ✅ Inliers preservados: {inlier_mask.sum()} ({(inlier_mask.sum()/len(X_embed)*100):.2f}%)")
    
    return inlier_mask


def train_viral_prediction_model(df_train):
    """
    Entrena modelo híbrido Autoencoder + XGBoost para predicción de viralidad.
    
    Flujo:
    1. Separa embeddings (384D) y features emocionales
    2. NUEVO: Limpia outliers con IQR + Isolation Forest
    3. Entrena autoencoder para reducir embeddings a 64D
    4. Combina features latentes (64D) con emocionales limpias
    5. Entrena XGBoost altamente regularizado
    6. Evalúa con validación cruzada estratificada
    
    Args:
        df_train (pd.DataFrame): Datos de entrenamiento
    """
    print("🚀 ENTRENANDO MODELO HÍBRIDO AUTOENCODER + XGBOOST")
    print("=" * 80)
    print("🎯 Objetivo: Predicir viralidad con embeddings reducidos + emociones")
    print("🧠 Pipeline: Embeddings(384D) -> Autoencoder(64D) -> XGBoost")
    print("📊 Validación: 5-fold cross-validation estratificada")
    print("=" * 80)
    
    # a. Preparar datos
    X_embed, X_emotion, y, emotion_names = prepare_features_for_hybrid_model(df_train)
    
    # === FASE DE LIMPIEZA DE OUTLIERS ===
    print("\n🧹 LIMPIEZA DE OUTLIERS")
    print("=" * 50)
    
    # 1. TRATAMIENTO IQR para X_emotion (Winsorization)
    print("📊 Aplicando Winsorization IQR a features emocionales...")
    X_emotion_clean = clean_emotion_features_iqr(X_emotion)
    
    # 2. TRATAMIENTO ISOLATION FOREST para X_embed
    print("🌲 Aplicando Isolation Forest a embeddings...")
    inlier_mask = detect_embedding_outliers(X_embed)
    
    # 3. FILTRADO FINAL
    print("🔍 Aplicando filtrado final con máscara de inliers...")
    X_embed = X_embed[inlier_mask]
    X_emotion = X_emotion_clean[inlier_mask]
    y = y[inlier_mask]
    
    print(f"   ✅ Datos después de limpieza:")
    print(f"      📊 Muestras restantes: {len(X_embed)} (eliminadas: {(~inlier_mask).sum()})")
    print(f"      🎯 Distribución viral: {y.sum()} virales / {len(y)} total")
    
    # Normalizar embeddings para el autoencoder
    print("\n📏 Normalizando embeddings...")
    scaler = StandardScaler()
    X_embed_scaled = scaler.fit_transform(X_embed)
    
    # b. Construir y entrenar autoencoder
    print("\n🔄 Entrenando Autoencoder...")
    autoencoder, encoder = build_autoencoder()
    
    # Entrenar autoencoder (40 epochs)
    print("   🎯 Entrenando por 40 epochs...")
    history = autoencoder.fit(
        X_embed_scaled, X_embed_scaled,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        shuffle=True
    )
    
    # c. Usar encoder para obtener representación latente
    print("\n🧬 Generando representación latente (64D)...")
    X_latent = encoder.predict(X_embed_scaled, verbose=0)
    print(f"   ✅ Features latentes: {X_latent.shape}")
    
    # d. Combinar features latentes con emocionales
    print("\n🔗 Combinando features latentes + emocionales...")
    X_final = np.concatenate([X_latent, X_emotion], axis=1)
    print(f"   ✅ Features finales: {X_final.shape} ({X_latent.shape[1]} latentes + {X_emotion.shape[1]} emocionales)")
    
    # e. Definir XGBoost altamente regularizado
    print("\n🤖 Configurando XGBoost altamente regularizado...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,           # Muchos árboles pero con learning rate bajo
        max_depth=4,                # Árboles poco profundos
        learning_rate=0.3,         # Learning rate muy conservador
        gamma=0.5,                  # Alta regularización (min split loss)
        min_child_weight=10,        # Evita hojas con pocas muestras
        subsample=0.7,              # Submuestreo de filas
        colsample_bytree=0.7,       # Submuestreo de columnas
        reg_alpha=0.1,              # Regularización L1
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1                   # Usar todos los cores
    )
    
    print("   ✅ XGBoost configurado con alta regularización")
    
    # f. Validación cruzada estratificada
    print("\n📊 Realizando validación cruzada estratificada (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calcular métricas
    cv_roc_auc = cross_val_score(xgb_model, X_final, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_f1 = cross_val_score(xgb_model, X_final, y, cv=cv, scoring='f1', n_jobs=-1)
    
    # g. Imprimir resultados
    print("\n🎯 RESULTADOS DE VALIDACIÓN CRUZADA:")
    print("=" * 50)
    print(f"📊 ROC-AUC: {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
    print(f"📊 F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    
    # Mostrar resultados por fold
    print(f"\n📋 Resultados detallados por fold:")
    for i, (roc, f1) in enumerate(zip(cv_roc_auc, cv_f1), 1):
        print(f"   Fold {i}: ROC-AUC={roc:.4f}, F1={f1:.4f}")
    
    # Entrenar modelo final para análisis
    print(f"\n🎯 Entrenando modelo final para análisis de features...")
    xgb_model.fit(X_final, y)
    
    # Analizar importancia de features
    analyze_feature_importance_hybrid(xgb_model, X_latent.shape[1], emotion_names)
    
    # Mostrar pérdida del autoencoder
    plot_autoencoder_loss(history)
    
    return {
        'cv_roc_auc': cv_roc_auc,
        'cv_f1': cv_f1,
        'autoencoder': autoencoder,
        'encoder': encoder,
        'xgb_model': xgb_model,
        'scaler': scaler
    }


def analyze_feature_importance_hybrid(model, n_latent_features, emotion_names):
    """
    Analiza la importancia de features en el modelo híbrido.
    
    Args:
        model: Modelo XGBoost entrenado
        n_latent_features (int): Número de features latentes
        emotion_names (list): Nombres de features emocionales
    """
    print(f"\n🔍 ANÁLISIS DE IMPORTANCIA DE FEATURES")
    print("=" * 50)
    
    importances = model.feature_importances_
    
    # Separar importancias
    latent_importances = importances[:n_latent_features]
    emotion_importances = importances[n_latent_features:]
    
    print(f"📊 Features Latentes (Autoencoder):")
    print(f"   🧬 Promedio: {latent_importances.mean():.4f}")
    print(f"   📈 Std: {latent_importances.std():.4f}")
    print(f"   🏆 Más importante: Latente-{latent_importances.argmax()} ({latent_importances.max():.4f})")
    
    print(f"\n🎭 Features Emocionales:")
    if len(emotion_importances) > 0:
        top_emotion_indices = emotion_importances.argsort()[-5:][::-1]
        print(f"   🏆 Top 5 features emocionales:")
        for i, idx in enumerate(top_emotion_indices, 1):
            if idx < len(emotion_names):
                print(f"      {i}. {emotion_names[idx]}: {emotion_importances[idx]:.4f}")
    
    # Comparación general
    latent_total = latent_importances.sum()
    emotion_total = emotion_importances.sum()
    total = latent_total + emotion_total
    
    print(f"\n⚖️ CONTRIBUCIÓN TOTAL:")
    print(f"   🧬 Features Latentes: {latent_total:.3f} ({latent_total/total*100:.1f}%)")
    print(f"   🎭 Features Emocionales: {emotion_total:.3f} ({emotion_total/total*100:.1f}%)")


def plot_autoencoder_loss(history):
    """
    Visualiza la pérdida del autoencoder durante el entrenamiento.
    
    Args:
        history: Historia del entrenamiento de Keras
    """
    plt.figure(figsize=(12, 4))
    
    # Pérdida de entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Autoencoder')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE (Mean Absolute Error)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE Entrenamiento')
    plt.plot(history.history['val_mae'], label='MAE Validación')
    plt.title('Error Absoluto Medio')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   💾 Gráfico guardado como 'autoencoder_training.png'")


def main():
    """Función principal que ejecuta todo el pipeline."""
    print("🎯 MODELO HÍBRIDO: AUTOENCODER + XGBOOST PARA PREDICCIÓN DE VIRALIDAD")
    print("=" * 80)
    print("🧠 Arquitectura: Embeddings(384D) -> Autoencoder(64D) + Emociones -> XGBoost")
    print("🎭 Features: Títulos limpios + análisis emocional completo")
    print("🔬 Validación: Cross-validation estratificada con alta regularización")
    print("=" * 80)
    
    # Cargar datos
    df_train = load_training_data_with_embeddings()
    
    if df_train.empty:
        print("❌ No se pudieron cargar los datos de entrenamiento")
        return
    
    # Entrenar modelo híbrido
    results = train_viral_prediction_model(df_train)
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("📊 Resultados:")
    print(f"   🎯 ROC-AUC medio: {results['cv_roc_auc'].mean():.4f}")
    print(f"   🎯 F1-Score medio: {results['cv_f1'].mean():.4f}")
    print("📁 Artefactos generados:")
    print("   🎨 autoencoder_training.png: Curvas de entrenamiento")
    print("   📊 Análisis de importancia de features")
    print("💡 El modelo combina representaciones latentes con análisis emocional")
    print("=" * 80)


if __name__ == '__main__':
    main()