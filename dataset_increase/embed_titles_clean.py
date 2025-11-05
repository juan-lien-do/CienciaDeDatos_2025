from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import re
import math

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Tamaño de lote para procesar títulos (ajusta según memoria/CPU)
BATCH_SIZE = 256
MODEL_NAME = 'all-MiniLM-L6-v2'


def clean_title(title: str) -> str:
    """
    Limpia el título eliminando la fuente (ej. "| Mashable") y espacios extra.
    
    Asume que la fuente está al final después de un separador común (como '|' o '-').
    """
    # 1. Eliminar todo lo que esté después de la primera barra vertical
    if '|' in title:
        title = title.split('|')[0]
    
    # 2. (Opcional) Limpiar separadores comunes al final
    title = re.sub(r'[\-—]\s*$', '', title).strip()

    # 3. Limpiar espacios en blanco al inicio/fin
    return title.strip()


def embed_clean_titles(batch_size=BATCH_SIZE):
    """Genera embeddings para títulos limpios y guarda en `embedding_titulo_clean`.

    Flujo:
    - Lee nodos que tengan `titulo` (streaming desde Neo4j)
    - Limpia títulos eliminando fuentes como "| Mashable"
    - Procesa en lotes: codifica con SentenceTransformer y actualiza en Neo4j
    - Guarda en nueva columna `embedding_titulo_clean` para no sobreescribir originales
    - Usa `id(n)` para referenciar y actualizar el mismo nodo sin perder otras propiedades
    """

    print("🧹 GENERACIÓN DE EMBEDDINGS PARA TÍTULOS LIMPIOS")
    print("=" * 60)
    print("🎯 Objetivo: Generar embeddings sin fuentes (ej. '| Mashable')")
    print("💾 Campo destino: embedding_titulo_clean")
    print("🧠 Modelo: all-MiniLM-L6-v2")
    print("=" * 60)

    print("\n🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Cargar modelo de embeddings una sola vez (costoso)
    print(f"🧠 Cargando modelo de embeddings '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    total_processed = 0
    total_updated = 0
    total_cleaned = 0  # Contador de títulos que realmente se limpiaron

    try:
        with driver.session() as session:
            # Query para obtener nodos que no tienen embedding_titulo_clean o necesitan actualización
            print("📊 Obteniendo nodos con 'titulo' para limpieza...")
            result = session.run("""
                MATCH (n) 
                WHERE n.titulo IS NOT NULL 
                  AND n.embedding_titulo_clean IS NULL
                RETURN id(n) AS nid, n.titulo AS titulo
            """)

            batch = []  # lista de tuples (nid, titulo_original, titulo_limpio)

            for record in result:
                nid = record["nid"]
                titulo_original = record["titulo"]

                # Validación de títulos nulos/vacíos
                if titulo_original is None or titulo_original == "":
                    continue
                
                # Normalizar a string
                titulo_text = str(titulo_original).strip()
                
                # Saltar si queda vacío después de limpiar o es muy corto
                if titulo_text == "" or len(titulo_text) < 3:
                    continue

                # 🧹 LIMPIAR TÍTULO
                titulo_limpio = clean_title(titulo_text)
                
                # Validar que el título limpio sigue siendo válido
                if len(titulo_limpio) < 3:
                    continue
                
                # Contar títulos que realmente cambiaron
                if titulo_limpio != titulo_text:
                    total_cleaned += 1

                batch.append((nid, titulo_text, titulo_limpio))

                # Si alcanzamos el tamaño de lote, procesamos
                if len(batch) >= batch_size:
                    updated = _process_and_update_clean_batch(session, model, batch)
                    total_processed += len(batch)
                    total_updated += updated
                    print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated}, limpiadas: {total_cleaned})")
                    batch = []

            # Procesar remanente
            if batch:
                updated = _process_and_update_clean_batch(session, model, batch)
                total_processed += len(batch)
                total_updated += updated
                print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated}, limpiadas: {total_cleaned})")

            # Mostrar estadísticas de limpieza
            print(f"\n📊 ESTADÍSTICAS DE LIMPIEZA:")
            print(f"   📈 Total procesados: {total_processed}")
            print(f"   💾 Total actualizados: {total_updated}")
            print(f"   🧹 Títulos modificados: {total_cleaned}")
            print(f"   📍 Títulos sin cambios: {total_processed - total_cleaned}")
            if total_processed > 0:
                clean_percentage = (total_cleaned / total_processed) * 100
                print(f"   📊 Porcentaje de limpieza: {clean_percentage:.1f}%")

    except Exception as e:
        print(f"❌ Error general durante el proceso: {e}")
        raise

    finally:
        driver.close()
        print(f"\n🔌 Conexión cerrada.")
        print(f"💾 Nuevos embeddings disponibles en: embedding_titulo_clean")


def _process_and_update_clean_batch(session, model, batch):
    """Codifica títulos limpios y actualiza los nodos en una sola transacción.

    Args:
        session: Sesión de Neo4j
        model: Modelo de SentenceTransformer
        batch: Lista de (nid, titulo_original, titulo_limpio)

    Devuelve la cantidad de nodos actualizados exitosamente.
    """
    nids = [item[0] for item in batch]
    clean_titles = [item[2] for item in batch]  # Usar títulos limpios para embeddings

    print(f"      🧹 Generando embeddings para {len(clean_titles)} títulos limpios...")
    
    # Mostrar ejemplos de limpieza (primeros 3)
    for i, (nid, original, clean) in enumerate(batch[:3]):
        if original != clean:
            print(f"         Ejemplo {i+1}: '{original[:50]}...' → '{clean[:50]}...'")

    # Generar embeddings en lote (devuelve numpy array)
    embeddings = model.encode(clean_titles, show_progress_bar=False)

    # Convertir embeddings a listas nativas de Python para Neo4j
    rows = []
    for nid, emb in zip(nids, embeddings):
        # emb puede ser numpy array; convertir a lista de floats
        emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))
        rows.append({"id": int(nid), "embedding_clean": emb_list})

    # Actualizar en una sola query usando UNWIND
    # NOTA: Guardamos en embedding_titulo_clean para no sobreescribir los originales
    update_query = (
        "UNWIND $rows AS row\n"
        "MATCH (n) WHERE id(n) = row.id\n"
        "SET n.embedding_titulo_clean = row.embedding_clean\n"
        "RETURN count(n) AS updated"
    )

    try:
        tx_result = session.run(update_query, rows=rows)
        rec = tx_result.single()
        updated = rec["updated"] if rec else 0
        return updated
    except Exception as e:
        print(f"❌ Error actualizando lote: {e}")
        # Como fallback intentar actualizar individualmente
        updated = 0
        for row in rows:
            try:
                session.run(
                    "MATCH (n) WHERE id(n) = $nid SET n.embedding_titulo_clean = $embedding_clean",
                    nid=row["id"], embedding_clean=row["embedding_clean"]
                )
                updated += 1
            except Exception as inner_e:
                print(f"   ⚠️ Error actualizando nodo {row['id']}: {inner_e}")
        return updated


def show_cleaning_examples():
    """Muestra ejemplos de limpieza de títulos desde Neo4j para validar."""
    
    print("\n🔍 EJEMPLOS DE LIMPIEZA DE TÍTULOS")
    print("=" * 50)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Obtener una muestra de títulos para mostrar el antes/después
            result = session.run("""
                MATCH (n) 
                WHERE n.titulo IS NOT NULL 
                RETURN n.titulo AS titulo
                LIMIT 10
            """)
            
            print("📝 Ejemplos de transformación:")
            print("-" * 50)
            
            for i, record in enumerate(result, 1):
                titulo_original = record["titulo"]
                titulo_limpio = clean_title(str(titulo_original))
                
                cambio = "✅ LIMPIADO" if titulo_original != titulo_limpio else "⚪ SIN CAMBIO"
                
                print(f"{i:2d}. {cambio}")
                print(f"     Original: {titulo_original}")
                print(f"     Limpio:   {titulo_limpio}")
                print()
            
    except Exception as e:
        print(f"❌ Error mostrando ejemplos: {e}")
    finally:
        driver.close()


def validate_cleaning_stats():
    """Muestra estadísticas de títulos que contienen separadores comunes."""
    
    print("\n📊 ESTADÍSTICAS DE TÍTULOS CON SEPARADORES")
    print("=" * 50)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Contar títulos con diferentes separadores
            separators = ['|', ' - ', ' — ', 'Mashable', 'mashable']
            
            total_result = session.run("MATCH (n) WHERE n.titulo IS NOT NULL RETURN count(n) AS total")
            total = total_result.single()["total"]
            
            print(f"📈 Total de títulos: {total}")
            print(f"📋 Análisis de separadores:")
            print("-" * 30)
            
            for sep in separators:
                if sep in ['Mashable', 'mashable']:
                    # Para estas buscar como substring
                    query = f"MATCH (n) WHERE n.titulo IS NOT NULL AND n.titulo CONTAINS '{sep}' RETURN count(n) AS count"
                else:
                    # Para separadores buscar literalmente
                    query = f"MATCH (n) WHERE n.titulo IS NOT NULL AND n.titulo CONTAINS '{sep}' RETURN count(n) AS count"
                
                result = session.run(query)
                count = result.single()["count"]
                percentage = (count / total) * 100 if total > 0 else 0
                
                print(f"   '{sep}': {count:6d} ({percentage:4.1f}%)")
            
    except Exception as e:
        print(f"❌ Error validando estadísticas: {e}")
    finally:
        driver.close()


if __name__ == '__main__':
    print("🧹 EMBEDDINGS DE TÍTULOS LIMPIOS")
    print("=" * 60)
    print("🎯 Genera embeddings sin fuentes (ej. '| Mashable')")
    print("💾 Guarda en: embedding_titulo_clean")
    print("🔄 No modifica embeddings originales")
    print("=" * 60)
    
    # Mostrar estadísticas previas
    validate_cleaning_stats()
    
    # Mostrar ejemplos de limpieza
    show_cleaning_examples()
    
    # Generar embeddings limpios
    embed_clean_titles(batch_size=BATCH_SIZE)
    
    print("\n" + "=" * 60)
    print("✅ PROCESO COMPLETADO")
    print("📊 Campos disponibles:")
    print("   🔸 embedding_titulo: Embeddings originales")
    print("   🔹 embedding_titulo_clean: Embeddings de títulos limpios")
    print("💡 Tip: Compara ambos en modelos para ver cuál funciona mejor")
    print("=" * 60)