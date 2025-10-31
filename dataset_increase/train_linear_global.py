from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def train_linear_global():
    print("🚀 Iniciando entrenamiento de Regresión Lineal Global")
    print("=" * 60)
    
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
            X_train = np.array(embeddings)
            y_train = np.array(shares)
            
            print(f"📐 Dimensiones X_train: {X_train.shape}")
            print(f"📊 Estadísticas de shares:")
            print(f"   - Min: {np.min(y_train):,}")
            print(f"   - Max: {np.max(y_train):,}")
            print(f"   - Media: {np.mean(y_train):.2f}")
            print(f"   - Mediana: {np.median(y_train):.2f}")
            
            # Escalar features con StandardScaler
            print("⚖️ Aplicando StandardScaler a X_train...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Aplicar transformación logarítmica al target
            print("📈 Aplicando transformación logarítmica a y_train...")
            y_train_log = np.log1p(y_train)
            
            print(f"📊 Estadísticas de y_train_log:")
            print(f"   - Min: {np.min(y_train_log):.4f}")
            print(f"   - Max: {np.max(y_train_log):.4f}")
            print(f"   - Media: {np.mean(y_train_log):.4f}")
            print(f"   - Mediana: {np.median(y_train_log):.4f}")
            
            # 3️⃣ Entrenamiento del modelo
            print("\n🧠 Entrenando modelo de Regresión Lineal...")
            training_start = time.time()
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_log)
            
            training_time = time.time() - training_start
            print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
            
            # Hacer predicciones en datos de entrenamiento para métricas
            print("🎯 Calculando métricas en datos de entrenamiento...")
            y_pred_log = model.predict(X_train_scaled)
            
            # Calcular métricas en escala logarítmica
            r2_log = r2_score(y_train_log, y_pred_log)
            mae_log = mean_absolute_error(y_train_log, y_pred_log)
            rmse_log = np.sqrt(mean_squared_error(y_train_log, y_pred_log))
            
            # Transformar predicciones de vuelta a escala original para métricas adicionales
            y_pred_original = np.expm1(y_pred_log)
            mae_original = mean_absolute_error(y_train, y_pred_original)
            rmse_original = np.sqrt(mean_squared_error(y_train, y_pred_original))
            
            # Mostrar métricas en consola
            print("\n📊 MÉTRICAS DEL MODELO:")
            print("=" * 40)
            print(f"🎯 R² (escala log): {r2_log:.6f}")
            print(f"📏 MAE (escala log): {mae_log:.6f}")
            print(f"📐 RMSE (escala log): {rmse_log:.6f}")
            print("=" * 40)
            print("📊 Métricas adicionales (escala original):")
            print(f"📏 MAE (shares): {mae_original:.2f}")
            print(f"📐 RMSE (shares): {rmse_original:.2f}")
            
            # 4️⃣ Guardar artefactos
            print("\n💾 Guardando artefactos...")
            
            # Guardar modelo
            model_filename = 'linear_regression_global.pkl'
            joblib.dump(model, model_filename)
            print(f"✅ Modelo guardado: {model_filename}")
            
            # Guardar scaler
            scaler_filename = 'scaler.pkl'
            joblib.dump(scaler, scaler_filename)
            print(f"✅ Scaler guardado: {scaler_filename}")
            
            # Crear DataFrame con métricas y guardarlo
            metrics_data = {
                'metric': ['R2_log', 'MAE_log', 'RMSE_log', 'MAE_original', 'RMSE_original'],
                'value': [r2_log, mae_log, rmse_log, mae_original, rmse_original],
                'n_samples': [n_samples] * 5,
                'n_features': [X_train.shape[1]] * 5,
                'training_time_seconds': [training_time] * 5
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            metrics_filename = 'train_metrics_global.csv'
            df_metrics.to_csv(metrics_filename, index=False)
            print(f"✅ Métricas guardadas: {metrics_filename}")
            
            # 5️⃣ Resumen final
            total_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("🎉 RESUMEN FINAL")
            print("=" * 60)
            print(f"📊 Número de muestras: {n_samples:,}")
            print(f"📐 Dimensiones de entrada: {X_train.shape}")
            print(f"🔧 Dimensiones escaladas: {X_train_scaled.shape}")
            print(f"📈 Target transformado: log1p(shares)")
            print(f"")
            print(f"🧠 Modelo: Regresión Lineal")
            print(f"⚖️ Preprocesamiento: StandardScaler")
            print(f"")
            print(f"📊 MÉTRICAS PRINCIPALES:")
            print(f"   🎯 R² (escala log): {r2_log:.6f}")
            print(f"   📏 MAE (escala log): {mae_log:.6f}")
            print(f"   📐 RMSE (escala log): {rmse_log:.6f}")
            print(f"")
            print(f"📊 MÉTRICAS INTERPRETABLES:")
            print(f"   📏 MAE (shares): {mae_original:.0f} shares")
            print(f"   📐 RMSE (shares): {rmse_original:.0f} shares")
            print(f"")
            print(f"⏱️ TIEMPOS:")
            print(f"   🧠 Entrenamiento: {training_time:.2f}s")
            print(f"   ⏰ Tiempo total: {total_time:.2f}s")
            print(f"")
            print(f"💾 ARCHIVOS GENERADOS:")
            print(f"   📦 {model_filename}")
            print(f"   ⚖️ {scaler_filename}")
            print(f"   📊 {metrics_filename}")
            print("=" * 60)
            
            # Información adicional del modelo
            print(f"\n🔍 INFORMACIÓN ADICIONAL DEL MODELO:")
            print(f"   📊 Coeficientes no nulos: {np.count_nonzero(model.coef_)}/{len(model.coef_)}")
            print(f"   📐 Intercepto: {model.intercept_:.6f}")
            print(f"   📈 Rango de coeficientes: [{np.min(model.coef_):.6f}, {np.max(model.coef_):.6f}]")
            
            # Distribución de predicciones
            pred_stats = {
                'min_pred': np.min(y_pred_original),
                'max_pred': np.max(y_pred_original),
                'mean_pred': np.mean(y_pred_original),
                'median_pred': np.median(y_pred_original)
            }
            
            print(f"\n📈 DISTRIBUCIÓN DE PREDICCIONES (escala original):")
            print(f"   📉 Mín predicción: {pred_stats['min_pred']:.0f} shares")
            print(f"   📈 Máx predicción: {pred_stats['max_pred']:.0f} shares")
            print(f"   📊 Media predicción: {pred_stats['mean_pred']:.0f} shares")
            print(f"   📍 Mediana predicción: {pred_stats['median_pred']:.0f} shares")
            
            print(f"\n✅ Entrenamiento completado exitosamente!")
            
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Conexión a Neo4j cerrada.")

if __name__ == "__main__":
    train_linear_global()