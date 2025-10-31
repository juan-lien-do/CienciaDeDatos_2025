from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def train_linear_poly_global():
    print("🚀 Iniciando entrenamiento de Regresión Lineal con Características Polinómicas")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1️⃣ Conexión a Neo4j
    print("\n🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Consultar noticias de entrenamiento
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
                
                if embedding is not None and share_count is not None:
                    embeddings.append(embedding)
                    shares.append(share_count)
            
            if not embeddings:
                print("❌ No se encontraron datos de entrenamiento.")
                return
            
            n_samples = len(embeddings)
            print(f"✅ Datos recolectados: {n_samples:,} muestras")
            
            # 2️⃣ Preparación de los datos
            print("\n🔧 Preparando datos...")
            
            # Convertir embeddings a matriz
            X = np.array(embeddings)
            y = np.array(shares)
            
            print(f"📐 Dimensiones X originales: {X.shape}")
            print(f"📊 Estadísticas de shares:")
            print(f"   - Min: {np.min(y):,}")
            print(f"   - Max: {np.max(y):,}")
            print(f"   - Media: {np.mean(y):.2f}")
            print(f"   - Mediana: {np.median(y):.2f}")
            print(f"   - Std: {np.std(y):.2f}")
            
            # Aplicar StandardScaler
            print("⚖️ Aplicando StandardScaler a X...")
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
            
            # 3️⃣ Generar características polinómicas
            print("\n🔄 Generando características polinómicas...")
            poly_start = time.time()
            
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(X_scaled)
            
            poly_time = time.time() - poly_start
            
            print(f"📐 Dimensiones ANTES de PolynomialFeatures: {X_scaled.shape}")
            print(f"📐 Dimensiones DESPUÉS de PolynomialFeatures: {X_poly.shape}")
            print(f"📈 Factor de expansión: {X_poly.shape[1] / X_scaled.shape[1]:.1f}x")
            print(f"⏱️ Tiempo generación polinómicas: {poly_time:.2f}s")
            
            # Mostrar uso de memoria aproximado
            memory_mb = (X_poly.nbytes / 1024 / 1024)
            print(f"💾 Memoria aproximada X_poly: {memory_mb:.1f} MB")
            
            # 4️⃣ Entrenamiento del modelo
            print("\n🧠 Entrenando modelo de Regresión Lineal con características polinómicas...")
            training_start = time.time()
            
            model = LinearRegression(n_jobs=-1)
            model.fit(X_poly, y_log)
            
            training_time = time.time() - training_start
            print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
            
            # 5️⃣ Hacer predicciones y calcular métricas
            print("🎯 Calculando métricas en datos de entrenamiento...")
            prediction_start = time.time()
            
            y_pred_log = model.predict(X_poly)
            prediction_time = time.time() - prediction_start
            
            print(f"⏱️ Tiempo de predicción: {prediction_time:.2f} segundos")
            
            # Calcular métricas en escala logarítmica
            r2_log = r2_score(y_log, y_pred_log)
            mae_log = mean_absolute_error(y_log, y_pred_log)
            rmse_log = np.sqrt(mean_squared_error(y_log, y_pred_log))
            
            # Transformar predicciones de vuelta a escala original
            y_pred_original = np.expm1(y_pred_log)
            mae_original = mean_absolute_error(y, y_pred_original)
            rmse_original = np.sqrt(mean_squared_error(y, y_pred_original))
            
            # Mostrar métricas en consola
            print("\n📊 MÉTRICAS DEL MODELO:")
            print("=" * 50)
            print("🎯 Métricas en escala logarítmica:")
            print(f"   📈 R² (escala log): {r2_log:.6f}")
            print(f"   📏 MAE (escala log): {mae_log:.6f}")
            print(f"   📐 RMSE (escala log): {rmse_log:.6f}")
            print("=" * 50)
            print("📊 Métricas en escala original:")
            print(f"   📏 MAE (shares): {mae_original:.2f}")
            print(f"   📐 RMSE (shares): {rmse_original:.2f}")
            print("=" * 50)
            
            # 6️⃣ Guardar artefactos
            print("\n💾 Guardando artefactos...")
            
            # Guardar modelo
            model_filename = 'linear_regression_poly_global.pkl'
            joblib.dump(model, model_filename)
            print(f"✅ Modelo guardado: {model_filename}")
            
            # Guardar scaler
            scaler_filename = 'scaler_poly.pkl'
            joblib.dump(scaler, scaler_filename)
            print(f"✅ Scaler guardado: {scaler_filename}")
            
            # Guardar generador de características polinómicas
            poly_filename = 'poly_features.pkl'
            joblib.dump(poly_features, poly_filename)
            print(f"✅ PolynomialFeatures guardado: {poly_filename}")
            
            # Crear DataFrame con métricas y guardarlo
            metrics_data = {
                'metric': ['R2_log', 'MAE_log', 'RMSE_log', 'MAE_original', 'RMSE_original'],
                'value': [r2_log, mae_log, rmse_log, mae_original, rmse_original],
                'n_samples': [n_samples] * 5,
                'original_features': [X.shape[1]] * 5,
                'poly_features': [X_poly.shape[1]] * 5,
                'polynomial_degree': [2] * 5,
                'training_time_seconds': [training_time] * 5,
                'poly_generation_time_seconds': [poly_time] * 5
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            metrics_filename = 'train_metrics_poly_global.csv'
            df_metrics.to_csv(metrics_filename, index=False)
            print(f"✅ Métricas guardadas: {metrics_filename}")
            
            # 7️⃣ Análisis de predicciones
            print("\n📈 Análisis de predicciones...")
            
            # Estadísticas de predicciones en escala original
            pred_stats = {
                'min_pred': np.min(y_pred_original),
                'max_pred': np.max(y_pred_original),
                'mean_pred': np.mean(y_pred_original),
                'median_pred': np.median(y_pred_original),
                'std_pred': np.std(y_pred_original)
            }
            
            print(f"📊 Distribución de predicciones (escala original):")
            print(f"   📉 Mín predicción: {pred_stats['min_pred']:.0f} shares")
            print(f"   📈 Máx predicción: {pred_stats['max_pred']:.0f} shares")
            print(f"   📊 Media predicción: {pred_stats['mean_pred']:.0f} shares")
            print(f"   📍 Mediana predicción: {pred_stats['median_pred']:.0f} shares")
            print(f"   📏 Std predicción: {pred_stats['std_pred']:.0f} shares")
            
            # Comparar con datos reales
            print(f"\n🔍 Comparación real vs predicción:")
            print(f"   📊 Correlación: {np.corrcoef(y, y_pred_original)[0,1]:.4f}")
            
            # Detectar predicciones negativas (problemáticas)
            negative_preds = np.sum(y_pred_original < 0)
            if negative_preds > 0:
                print(f"   ⚠️ Predicciones negativas: {negative_preds}/{len(y_pred_original)}")
            else:
                print(f"   ✅ Todas las predicciones son positivas")
            
            # 8️⃣ Resumen final
            total_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("🎉 RESUMEN FINAL - REGRESIÓN LINEAL CON CARACTERÍSTICAS POLINÓMICAS")
            print("=" * 80)
            print(f"📊 DATOS:")
            print(f"   🔢 Número de muestras: {n_samples:,}")
            print(f"   📐 Dimensiones originales: {X.shape}")
            print(f"   ⚖️ Dimensiones escaladas: {X_scaled.shape}")
            print(f"   🔄 Dimensiones polinómicas: {X_poly.shape}")
            print(f"   📈 Grado polinómico: 2")
            print(f"   📊 Target: log1p(shares)")
            print(f"")
            print(f"🧠 MODELO:")
            print(f"   🤖 Algoritmo: Regresión Lineal")
            print(f"   ⚖️ Preprocesamiento: StandardScaler + PolynomialFeatures(degree=2)")
            print(f"   🔧 Configuración: n_jobs=-1")
            print(f"")
            print(f"📊 MÉTRICAS PRINCIPALES:")
            print(f"   🎯 R² (escala log): {r2_log:.6f}")
            print(f"   📏 MAE (escala log): {mae_log:.6f}")
            print(f"   📐 RMSE (escala log): {rmse_log:.6f}")
            print(f"")
            print(f"📊 MÉTRICAS INTERPRETABLES:")
            print(f"   📏 MAE (shares): {mae_original:.0f} shares")
            print(f"   📐 RMSE (shares): {rmse_original:.0f} shares")
            print(f"   🔗 Correlación: {np.corrcoef(y, y_pred_original)[0,1]:.4f}")
            print(f"")
            print(f"⏱️ TIEMPOS DE PROCESAMIENTO:")
            print(f"   🔄 Generación polinómicas: {poly_time:.2f}s")
            print(f"   🧠 Entrenamiento modelo: {training_time:.2f}s")
            print(f"   🎯 Predicción: {prediction_time:.2f}s")
            print(f"   ⏰ Tiempo total: {total_time:.2f}s")
            print(f"")
            print(f"📈 RANGO DE PREDICCIONES:")
            print(f"   📉 Mínima: {pred_stats['min_pred']:.0f} shares")
            print(f"   📈 Máxima: {pred_stats['max_pred']:.0f} shares")
            print(f"   📊 Promedio: {pred_stats['mean_pred']:.0f} shares")
            print(f"")
            print(f"💾 ARCHIVOS GENERADOS:")
            print(f"   📦 {model_filename}")
            print(f"   ⚖️ {scaler_filename}")
            print(f"   🔄 {poly_filename}")
            print(f"   📊 {metrics_filename}")
            print("=" * 80)
            
            # Información adicional del modelo
            print(f"\n🔍 INFORMACIÓN ADICIONAL DEL MODELO:")
            print(f"   📊 Coeficientes totales: {len(model.coef_):,}")
            print(f"   📊 Coeficientes no nulos: {np.count_nonzero(model.coef_):,}")
            print(f"   📐 Intercepto: {model.intercept_:.6f}")
            print(f"   📈 Rango coeficientes: [{np.min(model.coef_):.6f}, {np.max(model.coef_):.6f}]")
            print(f"   💾 Memoria modelo: {(model.coef_.nbytes + 8) / 1024:.1f} KB")
            
            # Análisis de características polinómicas más importantes
            coef_abs = np.abs(model.coef_)
            top_features_idx = np.argsort(coef_abs)[-10:]  # Top 10 características
            
            print(f"\n🏆 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES (por |coeficiente|):")
            for i, idx in enumerate(reversed(top_features_idx), 1):
                print(f"   {i:2d}. Feature {idx:4d}: {model.coef_[idx]:+.6f}")
            
            print(f"\n✅ Entrenamiento con características polinómicas completado exitosamente!")
            
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión a Neo4j cerrada.")

if __name__ == "__main__":
    train_linear_poly_global()