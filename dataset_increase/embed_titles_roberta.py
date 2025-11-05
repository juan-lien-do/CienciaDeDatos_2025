from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import math

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Tamaño de lote para procesar títulos (ajusta según memoria/CPU)
# Nota: RoBERTa es más pesado que MiniLM, usar lotes más pequeños
BATCH_SIZE = 128
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'


def embed_titles_roberta(batch_size=BATCH_SIZE):
    """Genera embeddings para el campo `titulo` usando Twitter-roBERTa-base-emotion y guarda en `embedding_titulo_emotion`.

    Flujo:
    - Lee nodos que tengan `titulo` (streaming desde Neo4j)
    - Procesa en lotes: codifica con Twitter-roBERTa-base-emotion y actualiza en Neo4j
    - Usa `id(n)` para referenciar y actualizar el mismo nodo sin perder otras propiedades
    - Guarda los embeddings en la propiedad `embedding_titulo_emotion`
    
    Ventajas de Twitter-roBERTa-base-emotion:
    - Entrenado específicamente en datos de Twitter (textos cortos)
    - Especializado en detección de emociones (alegría, enojo, miedo, sorpresa, etc.)
    - Ideal para predecir viralidad (contenido emocional tiende a ser más viral)
    - Robustez mejorada para textos informales
    - Dimensionalidad: 768 (vs 384 de MiniLM)
    """

    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Cargar modelo de embeddings una sola vez (costoso)
    print(f"🤖 Cargando modelo Twitter-roBERTa-base-emotion '{MODEL_NAME}'...")
    print("   � Este modelo es más pesado pero específico para análisis de emociones")
    print("   🎯 Optimizado para detectar emociones en textos cortos (alegría, enojo, miedo, etc.)")
    print("   📈 Ideal para predecir viralidad (contenido emocional es más compartible)")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"   ✅ Modelo cargado exitosamente (dimensionalidad: {model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"   ❌ Error cargando modelo: {e}")
        print("   💡 Instalando modelo automáticamente...")
        # Intentar cargar con auto-download
        model = SentenceTransformer(MODEL_NAME)

    total_processed = 0
    total_updated = 0

    try:
        with driver.session() as session:
            # Verificar si ya existen embeddings de emociones
            print("🔍 Verificando embeddings de emociones existentes...")
            check_result = session.run(
                "MATCH (n) WHERE n.embedding_titulo_emotion IS NOT NULL RETURN count(n) AS existing_count"
            )
            existing_count = check_result.single()["existing_count"]
            print(f"   📊 Embeddings de emociones existentes: {existing_count}")
            
            # Ejecutar query que devuelve id(n) y titulo para todos los nodos que tienen titulo
            # pero no tienen embedding_titulo_emotion (evitar reprocessar)
            print("📊 Obteniendo nodos con 'titulo' sin embedding de emociones (streaming)...")
            result = session.run("""
                MATCH (n) 
                WHERE n.titulo IS NOT NULL 
                  AND n.embedding_titulo_emotion IS NULL
                RETURN id(n) AS nid, n.titulo AS titulo
            """)

            batch = []  # lista de tuples (nid, titulo)
            skipped_count = 0

            for record in result:
                nid = record["nid"]
                titulo = record["titulo"]

                # MEJORA: Validación más robusta de títulos nulos/vacíos
                # Saltar títulos nulos o vacíos completamente
                if titulo is None or titulo == "":
                    skipped_count += 1
                    continue
                
                # Normalizar a string y limpiar espacios en blanco
                titulo_text = str(titulo).strip()
                
                # Saltar si queda vacío después de limpiar o es muy corto
                if titulo_text == "" or len(titulo_text) < 3:
                    skipped_count += 1
                    continue

                batch.append((nid, titulo_text))

                # Si alcanzamos el tamaño de lote, procesamos
                if len(batch) >= batch_size:
                    updated = _process_and_update_batch_roberta(session, model, batch)
                    total_processed += len(batch)
                    total_updated += updated
                    print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated}, saltadas: {skipped_count})")
                    batch = []

            # Procesar remanente
            if batch:
                updated = _process_and_update_batch_roberta(session, model, batch)
                total_processed += len(batch)
                total_updated += updated
                print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated}, saltadas: {skipped_count})")

    except Exception as e:
        print(f"❌ Error general durante el proceso: {e}")
        raise

    finally:
        driver.close()
        print(f"\n🔌 Conexión cerrada.")
        print(f"📊 Resumen del procesamiento:")
        print(f"   📈 Total procesado: {total_processed}")
        print(f"   ✅ Total actualizado: {total_updated}")
        print(f"   🤖 Modelo usado: Twitter-roBERTa-base-emotion")
        print(f"   📏 Dimensionalidad: 768")
        print(f"   😊 Especialización: Detección de emociones para viralidad")


def _process_and_update_batch_roberta(session, model, batch):
    """Codifica una lista de (nid, titulo) con RoBERTa y actualiza los nodos en una sola transacción.

    Devuelve la cantidad de nodos actualizados exitosamente.
    """
    nids = [item[0] for item in batch]
    titles = [item[1] for item in batch]

    try:
        # Generar embeddings en lote con RoBERTa (devuelve numpy array)
        # RoBERTa puede ser más lento, mostrar progreso
        print(f"      🤖 Generando embeddings RoBERTa para {len(titles)} títulos...")
        embeddings = model.encode(titles, show_progress_bar=True, batch_size=32)  # Batch más pequeño para RoBERTa
        
        # Convertir embeddings a listas nativas de Python para Neo4j
        rows = []
        for nid, emb in zip(nids, embeddings):
            # emb puede ser numpy array; convertir a lista de floats
            emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))
            rows.append({"id": int(nid), "embedding": emb_list})

        # Actualizar en una sola query usando UNWIND
        # IMPORTANTE: Usar embedding_titulo_emotion como nombre de propiedad
        update_query = (
            "UNWIND $rows AS row\n"
            "MATCH (n) WHERE id(n) = row.id\n"
            "SET n.embedding_titulo_emotion = row.embedding\n"
            "RETURN count(n) AS updated"
        )

        tx_result = session.run(update_query, rows=rows)
        rec = tx_result.single()
        updated = rec["updated"] if rec else 0
        return updated
        
    except Exception as e:
        print(f"❌ Error actualizando lote RoBERTa: {e}")
        # Como fallback intentar actualizar individualmente para encontrar nodos problemáticos
        updated = 0
        for i, (nid, titulo) in enumerate(batch):
            try:
                # Generar embedding individual
                embedding = model.encode([titulo], show_progress_bar=False)[0]
                emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(map(float, embedding))
                
                session.run(
                    "MATCH (n) WHERE id(n) = $nid SET n.embedding_titulo_emotion = $embedding",
                    nid=int(nid), embedding=emb_list
                )
                updated += 1
            except Exception as inner_e:
                print(f"   ⚠️ Error actualizando nodo {nid} (título: '{titulo[:50]}...'): {inner_e}")
        return updated


def check_roberta_embeddings_stats():
    """
    Función auxiliar para verificar estadísticas de los embeddings de emociones generados.
    """
    print("🔍 Verificando estadísticas de embeddings de emociones...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Contar nodos con embeddings de emociones
            result = session.run("""
                MATCH (n) 
                WHERE n.embedding_titulo_emotion IS NOT NULL
                RETURN count(n) AS count_emotion, 
                       size(n.embedding_titulo_emotion[0..1]) AS sample_dimension
                LIMIT 1
            """)
            
            record = result.single()
            if record and record["count_emotion"] > 0:
                count = record["count_emotion"]
                print(f"   ✅ Nodos con embedding_titulo_emotion: {count}")
                
                # Verificar dimensionalidad
                dim_result = session.run("""
                    MATCH (n) 
                    WHERE n.embedding_titulo_emotion IS NOT NULL
                    RETURN size(n.embedding_titulo_emotion) AS dimension
                    LIMIT 1
                """)
                dim_record = dim_result.single()
                if dim_record:
                    dimension = dim_record["dimension"]
                    print(f"   📏 Dimensionalidad verificada: {dimension}")
                
                # Comparar con embeddings MiniLM si existen
                compare_result = session.run("""
                    MATCH (n) 
                    WHERE n.embedding_titulo IS NOT NULL 
                      AND n.embedding_titulo_emotion IS NOT NULL
                    RETURN count(n) AS both_embeddings
                """)
                both_count = compare_result.single()["both_embeddings"]
                print(f"   🔄 Nodos con ambos embeddings (MiniLM + Emotion): {both_count}")
                
            else:
                print(f"   ❌ No se encontraron embeddings de emociones")
                
    except Exception as e:
        print(f"❌ Error verificando estadísticas: {e}")
    
    finally:
        driver.close()


if __name__ == '__main__':
    print("🚀 Iniciando generación de embeddings con Twitter-roBERTa-base-emotion")
    print("=" * 75)
    print("🎯 Objetivo: Generar embeddings optimizados para análisis de emociones")
    print("� Modelo: Twitter-roBERTa-base-emotion (768 dimensiones)")
    print("� Especialización: Detección de emociones para predecir viralidad")
    print(" Campo destino: embedding_titulo_emotion")
    print("⚡ Batch size reducido por mayor complejidad del modelo")
    print("=" * 75)
    
    # Ajusta BATCH_SIZE si necesitas más/menos paralelismo o memoria
    embed_titles_roberta(batch_size=BATCH_SIZE)
    
    print("\n" + "=" * 75)
    check_roberta_embeddings_stats()
    print("=" * 75)
    print("✅ Proceso completado. Los embeddings de emociones están listos para análisis.")
    print("😊 Tip: Estos embeddings capturan mejor las emociones que pueden predecir viralidad")
    print("🔍 Campo generado: embedding_titulo_emotion (768 dimensiones)")