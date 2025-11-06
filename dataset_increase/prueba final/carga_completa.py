from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configuración
BATCH_SIZE = 100  # Procesar en lotes para evitar timeouts
PREFIX = "orig_"  # Prefijo para features originales para evitar conflictos

def normalizar_url(url):
    """
    Normaliza URLs para hacer matching consistente.
    
    Aplica múltiples transformaciones de normalización:
    - Elimina protocolo (http/https)
    - Convierte a lowercase
    - Elimina trailing slash
    - Elimina query parameters
    - Elimina fragments (#)
    - Limpia espacios y comillas
    """
    if pd.isna(url) or url == "":
        return ""
    
    try:
        url = str(url).strip().lower()
        
        # Remover comillas
        url = url.replace('"', '').replace("'", "")
        
        # Remover protocolo
        url = re.sub(r'^https?://', '', url)
        
        # Remover www. si existe
        url = re.sub(r'^www\.', '', url)
        
        # Parsear para limpiar query params y fragments
        if '?' in url:
            url = url.split('?')[0]
        if '#' in url:
            url = url.split('#')[0]
        
        # Remover trailing slash
        if url.endswith('/'):
            url = url[:-1]
        
        # Limpiar espacios extra
        url = url.strip()
        
        return url
        
    except Exception as e:
        print(f"   ⚠️ Error normalizando URL '{url}': {e}")
        return ""


def load_csv_data():
    """
    Carga y prepara los datos del CSV original.
    
    Returns:
        pd.DataFrame: Datos del CSV con URLs normalizadas
    """
    print("📊 Cargando dataset original OnlineNewsPopularity.csv...")
    
    try:
        # Leer CSV
        df = pd.read_csv('./OnlineNewsPopularity.csv')
        
        print(f"   ✅ Cargadas {len(df)} filas del dataset original")
        print(f"   📋 Columnas disponibles: {len(df.columns)}")
        
        # Normalizar URLs
        print("   🔧 Normalizando URLs...")
        df['url_norm'] = df['url'].apply(normalizar_url)
        
        # Filtrar URLs válidas
        df_valid = df[df['url_norm'] != ''].copy()
        invalid_count = len(df) - len(df_valid)
        
        if invalid_count > 0:
            print(f"   ⚠️ Eliminadas {invalid_count} filas con URLs inválidas")
        
        print(f"   ✅ Dataset preparado: {len(df_valid)} filas con URLs válidas")
        
        return df_valid
        
    except Exception as e:
        print(f"   ❌ Error cargando CSV: {e}")
        return pd.DataFrame()


def get_existing_urls_from_neo4j():
    """
    Obtiene las URLs existentes en Neo4j para hacer el cruce.
    
    Returns:
        set: Conjunto de URLs normalizadas existentes en Neo4j
    """
    print("\n🔗 Conectando a Neo4j para obtener URLs existentes...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    existing_urls = set()
    
    try:
        with driver.session() as session:
            # Obtener todas las URLs de noticias existentes
            result = session.run("MATCH (n:Noticia) RETURN n.url AS url")
            
            for record in result:
                original_url = record["url"]
                if original_url:
                    normalized = normalizar_url(original_url)
                    if normalized:
                        existing_urls.add(normalized)
            
            print(f"   ✅ Obtenidas {len(existing_urls)} URLs únicas de Neo4j")
            
    except Exception as e:
        print(f"   ❌ Error conectando a Neo4j: {e}")
        
    finally:
        driver.close()
    
    return existing_urls


def prepare_feature_columns(df):
    """
    Prepara las columnas de features con prefijos y validación.
    
    Args:
        df (pd.DataFrame): DataFrame con datos originales
        
    Returns:
        tuple: (feature_columns, excluded_columns)
    """
    print("\n🔧 Preparando columnas de features...")
    
    # Columnas a excluir del mapping (ya existen o no son features)
    exclude_columns = {'url', 'url_norm', 'shares'}  # shares ya existe
    
    # Obtener todas las columnas que serán features
    all_columns = set(df.columns)
    feature_columns = all_columns - exclude_columns
    
    print(f"   📊 Total de features a cargar: {len(feature_columns)}")
    print(f"   🚫 Columnas excluidas: {exclude_columns}")
    
    # Mostrar algunas features de ejemplo
    sample_features = list(feature_columns)[:10]
    print(f"   📋 Ejemplo de features: {sample_features}")
    
    return feature_columns, exclude_columns


def create_feature_mapping(feature_columns):
    """
    Crea el mapping de columnas originales a propiedades de Neo4j con prefijo.
    
    Args:
        feature_columns (set): Columnas de features
        
    Returns:
        dict: Mapping de columna_original -> propiedad_neo4j
    """
    print(f"\n🏷️ Creando mapping de features con prefijo '{PREFIX}'...")
    
    feature_mapping = {}
    
    for col in feature_columns:
        # Limpiar nombre de columna para Neo4j
        clean_col = col.strip().replace(' ', '_').replace('-', '_')
        
        # Aplicar prefijo
        neo4j_prop = f"{PREFIX}{clean_col}"
        
        feature_mapping[col] = neo4j_prop
    
    print(f"   ✅ Creado mapping para {len(feature_mapping)} features")
    
    # Mostrar algunos ejemplos
    sample_items = list(feature_mapping.items())[:5]
    for orig, neo4j_name in sample_items:
        print(f"      {orig} -> {neo4j_name}")
    
    return feature_mapping


def update_neo4j_features(df_matched, feature_mapping):
    """
    Actualiza Neo4j con las features del dataset original.
    
    Args:
        df_matched (pd.DataFrame): DataFrame con noticias que coinciden
        feature_mapping (dict): Mapping de columnas a propiedades Neo4j
    """
    print(f"\n💾 Actualizando {len(df_matched)} noticias en Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        total_updated = 0
        total_batches = (len(df_matched) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(0, len(df_matched), BATCH_SIZE):
            batch_df = df_matched.iloc[batch_idx:batch_idx + BATCH_SIZE]
            current_batch = (batch_idx // BATCH_SIZE) + 1
            
            print(f"   📦 Procesando lote {current_batch}/{total_batches} ({len(batch_df)} noticias)...")
            
            with driver.session() as session:
                for idx, row in batch_df.iterrows():
                    try:
                        # Construir query dinámicamente
                        set_clauses = []
                        params = {'url_norm': row['url_norm']}
                        
                        for orig_col, neo4j_prop in feature_mapping.items():
                            value = row[orig_col]
                            
                            # Manejar valores nulos/nan
                            if pd.isna(value):
                                value = None
                            elif isinstance(value, (np.integer, np.floating)):
                                # Convertir numpy types a tipos nativos de Python
                                if np.isnan(value):
                                    value = None
                                else:
                                    value = float(value) if isinstance(value, np.floating) else int(value)
                            
                            set_clauses.append(f"n.{neo4j_prop} = ${neo4j_prop}")
                            params[neo4j_prop] = value
                        
                        # Construir query completa - Compatible con Neo4j versiones anteriores
                        query = f"""
                            MATCH (n:Noticia)
                            WHERE n.url IS NOT NULL
                            WITH n, 
                                 CASE 
                                   WHEN left(toLower(n.url), 7) = 'http://' THEN right(toLower(n.url), size(toLower(n.url)) - 7)
                                   WHEN left(toLower(n.url), 8) = 'https://' THEN right(toLower(n.url), size(toLower(n.url)) - 8)
                                   ELSE toLower(n.url)
                                 END AS normalized_url
                            WHERE replace(replace(replace(normalized_url, 'www.', ''), '/', ''), 'www.', '') = replace(replace($url_norm, '/', ''), 'www.', '')
                            SET {', '.join(set_clauses)}
                            RETURN count(n) AS updated
                        """
                        
                        result = session.run(query, **params)
                        updated = result.single()["updated"]
                        total_updated += updated
                        
                        if updated == 0:
                            print(f"      ⚠️ No se encontró noticia para URL: {row['url_norm'][:50]}...")
                        
                    except Exception as e:
                        print(f"      ❌ Error actualizando fila {idx}: {e}")
                        continue
            
            print(f"      ✅ Lote {current_batch} completado")
        
        print(f"\n🎯 ACTUALIZACIÓN COMPLETADA:")
        print(f"   📊 Total de noticias actualizadas: {total_updated}")
        print(f"   📈 Features agregadas por noticia: {len(feature_mapping)}")
        print(f"   🏷️ Prefijo usado: '{PREFIX}'")
        
    except Exception as e:
        print(f"   ❌ Error durante actualización: {e}")
        
    finally:
        driver.close()


def verify_feature_loading():
    """
    Verifica que las features se hayan cargado correctamente.
    """
    print(f"\n🔍 Verificando carga de features...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Contar noticias con features originales
            result = session.run(f"""
                MATCH (n:Noticia)
                WHERE n.{PREFIX}timedelta IS NOT NULL
                RETURN count(n) AS count_with_features
            """)
            
            count_with_features = result.single()["count_with_features"]
            
            # Contar total de noticias
            result_total = session.run("MATCH (n:Noticia) RETURN count(n) AS total")
            total_noticias = result_total.single()["total"]
            
            print(f"   📊 Noticias con features originales: {count_with_features}")
            print(f"   📊 Total de noticias: {total_noticias}")
            
            if count_with_features > 0:
                percentage = (count_with_features / total_noticias) * 100
                print(f"   📈 Porcentaje con features: {percentage:.1f}%")
                
                # Mostrar ejemplo de feature
                sample_result = session.run(f"""
                    MATCH (n:Noticia)
                    WHERE n.{PREFIX}timedelta IS NOT NULL
                    RETURN n.url AS url, n.{PREFIX}timedelta AS timedelta, n.{PREFIX}shares AS orig_shares
                    LIMIT 3
                """)
                
                print(f"   📋 Ejemplos de features cargadas:")
                for record in sample_result:
                    print(f"      URL: {record['url'][:50]}...")
                    print(f"      Timedelta: {record['timedelta']}")
                    print(f"      Shares originales: {record['orig_shares']}")
                    print()
            
    except Exception as e:
        print(f"   ❌ Error en verificación: {e}")
        
    finally:
        driver.close()


def main():
    """
    Función principal que ejecuta todo el proceso de carga.
    """
    print("🚀 CARGA COMPLETA DE FEATURES ORIGINALES A NEO4J")
    print("=" * 80)
    print("🎯 Objetivo: Cruzar dataset original con Neo4j y cargar features faltantes")
    print(f"🏷️ Prefijo para features: '{PREFIX}'")
    print("🔗 Matching: Por URL normalizada")
    print("=" * 80)
    
    # 1. Cargar datos del CSV
    df_csv = load_csv_data()
    if df_csv.empty:
        print("❌ No se pudieron cargar datos del CSV")
        return
    
    # 2. Obtener URLs existentes en Neo4j
    existing_urls = get_existing_urls_from_neo4j()
    if not existing_urls:
        print("❌ No se encontraron URLs en Neo4j")
        return
    
    # 3. Hacer cruce por URL normalizada
    print(f"\n🔗 Cruzando datos por URL normalizada...")
    df_matched = df_csv[df_csv['url_norm'].isin(existing_urls)].copy()
    
    print(f"   ✅ Matches encontrados: {len(df_matched)}")
    print(f"   📊 Porcentaje de match: {(len(df_matched)/len(df_csv))*100:.1f}%")
    
    if len(df_matched) == 0:
        print("❌ No se encontraron coincidencias entre CSV y Neo4j")
        return
    
    # 4. Preparar features
    feature_columns, excluded = prepare_feature_columns(df_matched)
    feature_mapping = create_feature_mapping(feature_columns)
    
    # 5. Actualizar Neo4j
    update_neo4j_features(df_matched, feature_mapping)
    
    # 6. Verificar carga
    verify_feature_loading()
    
    print("\n" + "=" * 80)
    print("✅ PROCESO DE CARGA COMPLETADO")
    print("📊 Resumen:")
    print(f"   🔢 Features cargadas: {len(feature_mapping)}")
    print(f"   📰 Noticias actualizadas: {len(df_matched)}")
    print(f"   🏷️ Prefijo usado: '{PREFIX}'")
    print("💡 Las features originales están ahora disponibles en Neo4j")
    print("🚀 Listo para análisis avanzados con datos completos")
    print("=" * 80)


if __name__ == '__main__':
    main()