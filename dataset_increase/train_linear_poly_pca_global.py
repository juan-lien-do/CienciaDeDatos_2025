from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def train_linear_poly_pca_global():
    print("🚀 Iniciando entrenamiento de Regresión Lineal con PCA + Características Polinómicas")
    print("=" * 85)
    
    start_time = time.time()
    
    # 1️⃣ Conexión a Neo4j
    print("\n🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # 📊 Consultar noticias de entrenamiento
            print("📊 Consultando noticias de entrenamiento...")
            result = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train'
                RETURN n.embedding AS embedding, n.shares AS shares
            """)
            
            # Recolectar datos
            embeddings = []
            shares = []
            
            print("📥 Recolectando datos...")
            for record in result:
                embedding = record["embedding"]
                share_count = record["shares"]
                
                # Descartar valores nulos
                if embedding is not None and share_count is not None:
                    embeddings.append(embedding)
                    shares.append(share_count)
            
            if not embeddings:
                print("❌ No se encontraron datos de entrenamiento.")
                return
            
            n_samples = len(embeddings)
            print(f"✅ Datos recolectados: {n_samples:,} muestras")
            
            # 2️⃣ Conversión a arrays NumPy
            print("\n🔧 Convirtiendo a arrays NumPy...")
            X = np.array(embeddings)
            y = np.array(shares)
            
            print(f"📐 Dimensiones X originales: {X.shape}")
            print(f"📊 Estadísticas de shares:")
            print(f"   - Min: {np.min(y):,}")
            print(f"   - Max: {np.max(y):,}")
            print(f"   - Media: {np.mean(y):.2f}")
            print(f"   - Mediana: {np.median(y):.2f}")
            print(f"   - Std: {np.std(y):.2f}")
            
            # 3️⃣ Preparación de datos
            print("\n⚙️ Escalando datos...")
            
            # Aplicar StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print(f"📐 Dimensiones X_scaled: {X_scaled.shape}")
            
            # Aplicar transformación logarítmica al target
            print("📈 Aplicando transformación logarítmica a y...")
            y_log = np.log1p(y)
            
            print(f"📊 Estadísticas de y_log:")
            print(f"   - Min: {np.min(y_log):.4f}")
            print(f"   - Max: {np.max(y_log):.4f}")
            print(f"   - Media: {np.mean(y_log):.4f}")
            print(f"   - Mediana: {np.median(y_log):.4f}")
            print(f"   - Std: {np.std(y_log):.4f}")
            
            # 4️⃣ Reducción de dimensionalidad con PCA
            print("\n📉 Aplicando PCA para reducción de dimensionalidad...")
            pca_start = time.time()
            
            pca = PCA(n_components=50, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            pca_time = time.time() - pca_start
            
            # Calcular varianza explicada acumulada
            explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
            total_explained_variance = explained_variance_cumsum[-1]
            
            print(f"📐 Dimensiones ANTES de PCA: {X_scaled.shape}")
            print(f"📐 Dimensiones DESPUÉS de PCA: {X_pca.shape}")
            print(f"📊 Varianza explicada acumulada: {total_explained_variance:.4f} ({total_explained_variance*100:.2f}%)")
            print(f"📈 Reducción dimensional: {X.shape[1]} → {X_pca.shape[1]} features")
            print(f"⏱️ Tiempo PCA: {pca_time:.2f}s")
            
            # Mostrar varianza explicada por componente (primeros 10)
            print(f"📊 Top 10 componentes PCA (varianza explicada individual):")
            for i in range(min(10, len(pca.explained_variance_ratio_))):
                var_ratio = pca.explained_variance_ratio_[i]
                print(f"   PC{i+1:2d}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            
            # 5️⃣ Expandir características polinómicas
            print("\n🔄 Expandiendo características polinómicas...")
            poly_start = time.time()
            
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(X_pca)
            
            poly_time = time.time() - poly_start
            
            print(f"📐 Dimensiones ANTES de PolynomialFeatures: {X_pca.shape}")
            print(f"📐 Dimensiones DESPUÉS de PolynomialFeatures: {X_poly.shape}")
            print(f"📈 Factor de expansión polinómica: {X_poly.shape[1] / X_pca.shape[1]:.1f}x")
            print(f"⏱️ Tiempo generación polinómicas: {poly_time:.2f}s")
            
            # Mostrar uso de memoria
            memory_mb = (X_poly.nbytes / 1024 / 1024)
            print(f"💾 Memoria aproximada X_poly: {memory_mb:.1f} MB")
            
            # 6️⃣ Entrenamiento del modelo lineal
            print("\n🎯 Entrenando modelo polinómico global...")
            training_start = time.time()
            
            model = LinearRegression(n_jobs=-1)
            model.fit(X_poly, y_log)
            
            training_time = time.time() - training_start
            print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
            
            # 7️⃣ Calcular métricas en datos de entrenamiento
            print("🎯 Calculando métricas en datos de entrenamiento...")
            prediction_start = time.time()
            
            y_pred_log = model.predict(X_poly)
            prediction_time = time.time() - prediction_start
            
            print(f"⏱️ Tiempo de predicción: {prediction_time:.2f} segundos")
            
            # Métricas en escala logarítmica
            r2_log = r2_score(y_log, y_pred_log)
            mae_log = mean_absolute_error(y_log, y_pred_log)
            rmse_log = np.sqrt(mean_squared_error(y_log, y_pred_log))
            
            # Convertir predicciones a escala original
            y_pred_original = np.expm1(y_pred_log)
            mae_original = mean_absolute_error(y, y_pred_original)
            rmse_original = np.sqrt(mean_squared_error(y, y_pred_original))
            
            # Correlación
            correlation = np.corrcoef(y, y_pred_original)[0,1]
            
            # 8️⃣ Mostrar resumen tabulado
            print("\n📊 RESUMEN TABULADO DEL MODELO:")
            print("=" * 70)
            
            # Tabla de dimensiones
            print("📐 DIMENSIONES:")
            print(f"{'Etapa':<25} {'Dimensiones':<15} {'Descripción':<25}")
            print("-" * 65)
            print(f"{'Originales':<25} {str(X.shape):<15} {'Embeddings raw':<25}")
            print(f"{'Escaladas':<25} {str(X_scaled.shape):<15} {'StandardScaler':<25}")
            print(f"{'PCA':<25} {str(X_pca.shape):<15} {'50 componentes':<25}")
            print(f"{'Polinómicas':<25} {str(X_poly.shape):<15} {'Grado 2':<25}")
            
            # Tabla de métricas
            print(f"\n📊 MÉTRICAS:")
            print(f"{'Métrica':<20} {'Valor':<15} {'Escala':<15}")
            print("-" * 50)
            print(f"{'Varianza PCA':<20} {total_explained_variance:<15.4f} {'Porcentaje':<15}")
            print(f"{'R²':<20} {r2_log:<15.6f} {'Log':<15}")
            print(f"{'MAE':<20} {mae_log:<15.6f} {'Log':<15}")
            print(f"{'RMSE':<20} {rmse_log:<15.6f} {'Log':<15}")
            print(f"{'MAE':<20} {mae_original:<15.2f} {'Shares':<15}")
            print(f"{'RMSE':<20} {rmse_original:<15.2f} {'Shares':<15}")
            print(f"{'Correlación':<20} {correlation:<15.4f} {'Pearson':<15}")
            
            # Tabla de tiempos
            total_time = time.time() - start_time
            print(f"\n⏱️ TIEMPOS DE PROCESAMIENTO:")
            print(f"{'Proceso':<25} {'Tiempo (s)':<15}")
            print("-" * 40)
            print(f"{'PCA':<25} {pca_time:<15.2f}")
            print(f"{'Polinómicas':<25} {poly_time:<15.2f}")
            print(f"{'Entrenamiento':<25} {training_time:<15.2f}")
            print(f"{'Predicción':<25} {prediction_time:<15.2f}")
            print(f"{'Total':<25} {total_time:<15.2f}")
            print("=" * 70)
            
            # 9️⃣ Guardar artefactos
            print("\n💾 Guardando artefactos...")
            
            # Guardar modelo
            model_filename = 'linear_regression_poly_pca.pkl'
            joblib.dump(model, model_filename)
            print(f"✅ Modelo guardado: {model_filename}")
            
            # Guardar escalador
            scaler_filename = 'scaler.pkl'
            joblib.dump(scaler, scaler_filename)
            print(f"✅ Scaler guardado: {scaler_filename}")
            
            # Guardar modelo PCA
            pca_filename = 'pca.pkl'
            joblib.dump(pca, pca_filename)
            print(f"✅ PCA guardado: {pca_filename}")
            
            # Guardar generador de características polinómicas
            poly_filename = 'poly_features.pkl'
            joblib.dump(poly_features, poly_filename)
            print(f"✅ PolynomialFeatures guardado: {poly_filename}")
            
            # Crear y guardar métricas en CSV
            metrics_data = {
                'metric': ['R2_log', 'MAE_log', 'RMSE_log', 'MAE_original', 'RMSE_original', 'correlation', 'pca_variance_explained'],
                'value': [r2_log, mae_log, rmse_log, mae_original, rmse_original, correlation, total_explained_variance],
                'n_samples': [n_samples] * 7,
                'original_features': [X.shape[1]] * 7,
                'pca_components': [X_pca.shape[1]] * 7,
                'poly_features': [X_poly.shape[1]] * 7,
                'polynomial_degree': [2] * 7,
                'pca_time_seconds': [pca_time] * 7,
                'poly_time_seconds': [poly_time] * 7,
                'training_time_seconds': [training_time] * 7
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            metrics_filename = 'train_metrics_poly_pca_global.csv'
            df_metrics.to_csv(metrics_filename, index=False)
            print(f"✅ Métricas guardadas: {metrics_filename}")
            
            # 🔟 Análisis adicional
            print(f"\n🔍 ANÁLISIS ADICIONAL:")
            
            # Estadísticas de predicciones
            pred_stats = {
                'min_pred': np.min(y_pred_original),
                'max_pred': np.max(y_pred_original),
                'mean_pred': np.mean(y_pred_original),
                'median_pred': np.median(y_pred_original)
            }
            
            print(f"📈 Rango de predicciones (escala original):")
            print(f"   📉 Mínima: {pred_stats['min_pred']:.0f} shares")
            print(f"   📈 Máxima: {pred_stats['max_pred']:.0f} shares")
            print(f"   📊 Promedio: {pred_stats['mean_pred']:.0f} shares")
            print(f"   📍 Mediana: {pred_stats['median_pred']:.0f} shares")
            
            # Información del modelo
            print(f"\n🔍 Información del modelo:")
            print(f"   📊 Coeficientes totales: {len(model.coef_):,}")
            print(f"   📊 Coeficientes no nulos: {np.count_nonzero(model.coef_):,}")
            print(f"   📐 Intercepto: {model.intercept_:.6f}")
            print(f"   📈 Rango coeficientes: [{np.min(model.coef_):.6f}, {np.max(model.coef_):.6f}]")
            
            # Verificar predicciones negativas
            negative_preds = np.sum(y_pred_original < 0)
            if negative_preds > 0:
                print(f"   ⚠️ Predicciones negativas: {negative_preds}/{len(y_pred_original)}")
            else:
                print(f"   ✅ Todas las predicciones son positivas")
            
            print(f"\n🎉 RESUMEN EJECUTIVO:")
            print(f"   🔢 Muestras procesadas: {n_samples:,}")
            print(f"   📐 Reducción dimensional: {X.shape[1]} → {X_pca.shape[1]} → {X_poly.shape[1]}")
            print(f"   📊 PCA varianza explicada: {total_explained_variance*100:.2f}%")
            print(f"   🎯 R² del modelo: {r2_log:.4f}")
            print(f"   📏 Error promedio: {mae_original:.0f} shares")
            print(f"   🔗 Correlación: {correlation:.4f}")
            print(f"   ⏰ Tiempo total: {total_time:.2f}s")
            
            print(f"\n💾 Archivos generados:")
            print(f"   📦 {model_filename}")
            print(f"   ⚖️ {scaler_filename}")
            print(f"   📉 {pca_filename}")
            print(f"   🔄 {poly_filename}")
            print(f"   📊 {metrics_filename}")
            
            print(f"\n✅ Entrenamiento completado exitosamente!")
            
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión a Neo4j cerrada.")

if __name__ == "__main__":
    train_linear_poly_pca_global()