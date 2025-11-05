from neo4j import GraphDatabase
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configuración del análisis de clusters
K_MIN = 5    # Número mínimo de clusters a probar
K_MAX = 100  # Número máximo de clusters a probar
K_STEP = 5   # Paso entre valores de k


def analyze_optimal_clusters():
    """
    Analiza el número óptimo de clusters usando título embeddings de Neo4j.
    
    Realiza:
    1. Carga embeddings de títulos desde Neo4j
    2. Prueba diferentes valores de k con KMeans
    3. Calcula inercia y silhouette score para cada k
    4. Genera gráficos del método del codo y silhouette
    5. Recomienda el k óptimo
    """
    
    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # 1️⃣ Cargar embeddings de títulos desde Neo4j
        print("📊 Cargando embeddings de títulos...")
        embeddings = load_title_embeddings(driver)
        
        if embeddings is None or len(embeddings) == 0:
            print("❌ No se encontraron embeddings de títulos. Ejecuta primero embed_titles.py")
            return
        
        print(f"✅ Cargados {len(embeddings)} embeddings de títulos")
        print(f"📐 Dimensiones: {embeddings.shape}")
        
        # 2️⃣ Probar diferentes números de clusters
        k_range = range(K_MIN, K_MAX + 1, K_STEP)
        print(f"🔍 Probando k desde {K_MIN} hasta {K_MAX} (paso {K_STEP})")
        
        results = analyze_k_range(embeddings, k_range)
        
        # 3️⃣ Mostrar resultados en tabla
        print_results_table(results)
        
        # 4️⃣ Encontrar el mejor k según silhouette score
        best_k = find_best_k(results)
        
        # 5️⃣ Generar gráficos
        plot_analysis_results(results, best_k)
        
        print(f"\n✅ Análisis completado. Gráficos guardados como 'elbow_method.png' y 'silhouette_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("🔌 Conexión cerrada.")


def load_title_embeddings(driver):
    """
    Carga todos los embeddings de títulos desde Neo4j.
    
    Returns:
        numpy.ndarray: Array con embeddings de títulos
    """
    with driver.session() as session:
        # Query para obtener embeddings de títulos
        result = session.run("""
            MATCH (n) 
            WHERE n.embedding_titulo IS NOT NULL 
            RETURN n.embedding_titulo AS embedding
        """)
        
        embeddings_list = []
        for record in result:
            embedding = record["embedding"]
            if embedding is not None:
                embeddings_list.append(embedding)
        
        if not embeddings_list:
            return None
        
        # Convertir a numpy array
        return np.array(embeddings_list)


def analyze_k_range(embeddings, k_range):
    """
    Analiza diferentes valores de k calculando inercia y silhouette score.
    
    Args:
        embeddings (numpy.ndarray): Array de embeddings
        k_range (range): Rango de valores k a probar
    
    Returns:
        list: Lista de diccionarios con resultados por k
    """
    results = []
    
    print("\n🧠 Entrenando modelos KMeans...")
    
    # Usar tqdm para mostrar progreso
    for k in tqdm(k_range, desc="Probando valores de k", unit="clusters"):
        try:
            # Entrenar KMeans con buenas prácticas
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init='auto',  # Usa el valor por defecto optimizado
                max_iter=300
            )
            
            # Ajustar el modelo
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calcular métricas
            inertia = kmeans.inertia_
            
            # Silhouette score (solo si k > 1 y k < n_samples)
            if k > 1 and k < len(embeddings):
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
            else:
                silhouette_avg = -1  # Valor inválido
            
            # Guardar resultados
            results.append({
                'k': k,
                'inertia': inertia,
                'silhouette_score': silhouette_avg,
                'n_samples': len(embeddings)
            })
            
        except Exception as e:
            print(f"\n⚠️ Error con k={k}: {e}")
            # Continuar con el siguiente valor de k
            continue
    
    return results


def print_results_table(results):
    """
    Imprime una tabla formateada con los resultados del análisis.
    
    Args:
        results (list): Lista de resultados por k
    """
    print("\n📊 RESULTADOS DEL ANÁLISIS DE CLUSTERS")
    print("=" * 60)
    print(f"{'K':<6} {'Inercia':<15} {'Silhouette':<12} {'Estado':<10}")
    print("=" * 60)
    
    for result in results:
        k = result['k']
        inertia = result['inertia']
        silhouette = result['silhouette_score']
        
        # Formatear silhouette score
        if silhouette > 0:
            sil_str = f"{silhouette:.4f}"
            status = "✅ Válido"
        else:
            sil_str = "N/A"
            status = "⚠️ Inválido"
        
        print(f"{k:<6} {inertia:<15.2f} {sil_str:<12} {status:<10}")
    
    print("=" * 60)


def find_best_k(results):
    """
    Encuentra el mejor k basado en el silhouette score máximo.
    
    Args:
        results (list): Lista de resultados por k
    
    Returns:
        int: Mejor valor de k
    """
    # Filtrar resultados válidos (silhouette > 0)
    valid_results = [r for r in results if r['silhouette_score'] > 0]
    
    if not valid_results:
        print("❌ No se encontraron resultados válidos para silhouette score")
        return None
    
    # Encontrar el k con mejor silhouette score
    best_result = max(valid_results, key=lambda x: x['silhouette_score'])
    best_k = best_result['k']
    best_silhouette = best_result['silhouette_score']
    
    print(f"\n🏆 MEJOR K SEGÚN SILHOUETTE SCORE:")
    print(f"   📊 K óptimo: {best_k}")
    print(f"   📈 Silhouette Score: {best_silhouette:.4f}")
    print(f"   📉 Inercia: {best_result['inertia']:.2f}")
    
    return best_k


def plot_analysis_results(results, best_k=None):
    """
    Genera gráficos del método del codo y análisis de silhouette.
    
    Args:
        results (list): Lista de resultados por k
        best_k (int, optional): Mejor k para resaltar en gráficos
    """
    # Extraer datos para gráficos
    k_values = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    silhouette_scores = [r['silhouette_score'] if r['silhouette_score'] > 0 else np.nan 
                        for r in results]
    
    # Configurar estilo de gráficos
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 📈 Gráfico 1: Método del Codo (K vs Inercia)
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inercia', fontsize=12)
    ax1.set_title('Método del Codo\n(K vs Inercia)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Resaltar mejor k si existe
    if best_k and best_k in k_values:
        best_idx = k_values.index(best_k)
        ax1.plot(best_k, inertias[best_idx], 'ro', markersize=12, 
                label=f'Mejor k={best_k}')
        ax1.legend()
    
    # 📊 Gráfico 2: Análisis de Silhouette (K vs Silhouette Score)
    # Filtrar NaN values para el gráfico
    valid_indices = [i for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
    valid_k = [k_values[i] for i in valid_indices]
    valid_scores = [silhouette_scores[i] for i in valid_indices]
    
    if valid_k:
        ax2.plot(valid_k, valid_scores, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Análisis de Silhouette\n(K vs Silhouette Score)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Resaltar mejor k
        if best_k and best_k in valid_k:
            best_idx = valid_k.index(best_k)
            ax2.plot(best_k, valid_scores[best_idx], 'ro', markersize=12,
                    label=f'Mejor k={best_k} (Score: {valid_scores[best_idx]:.4f})')
            ax2.legend()
    
    # Ajustar layout y guardar
    plt.tight_layout()
    
    # Guardar gráficos individuales
    fig1, ax1_copy = plt.subplots(figsize=(8, 6))
    ax1_copy.plot(k_values, inertias, 'bo-', linewidth=2, markersize=6)
    ax1_copy.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1_copy.set_ylabel('Inercia', fontsize=12)
    ax1_copy.set_title('Método del Codo (K vs Inercia)', fontsize=14, fontweight='bold')
    ax1_copy.grid(True, alpha=0.3)
    if best_k and best_k in k_values:
        best_idx = k_values.index(best_k)
        ax1_copy.plot(best_k, inertias[best_idx], 'ro', markersize=12, 
                     label=f'Mejor k={best_k}')
        ax1_copy.legend()
    fig1.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    if valid_k:
        fig2, ax2_copy = plt.subplots(figsize=(8, 6))
        ax2_copy.plot(valid_k, valid_scores, 'go-', linewidth=2, markersize=6)
        ax2_copy.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax2_copy.set_ylabel('Silhouette Score', fontsize=12)
        ax2_copy.set_title('Análisis de Silhouette (K vs Silhouette Score)', fontsize=14, fontweight='bold')
        ax2_copy.grid(True, alpha=0.3)
        if best_k and best_k in valid_k:
            best_idx = valid_k.index(best_k)
            ax2_copy.plot(best_k, valid_scores[best_idx], 'ro', markersize=12,
                         label=f'Mejor k={best_k} (Score: {valid_scores[best_idx]:.4f})')
            ax2_copy.legend()
        fig2.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Mostrar gráfico combinado
    plt.show()
    
    print(f"\n📁 Gráficos guardados:")
    print(f"   📈 elbow_method.png - Método del codo")
    print(f"   📊 silhouette_analysis.png - Análisis de silhouette")


if __name__ == "__main__":
    print("🚀 Iniciando análisis de número óptimo de clusters")
    print("=" * 60)
    print(f"📊 Rango de k: {K_MIN} a {K_MAX} (paso {K_STEP})")
    print(f"🧠 Usando embeddings de títulos desde Neo4j")
    print(f"📈 Métricas: Inercia + Silhouette Score")
    print("=" * 60)
    
    analyze_optimal_clusters()