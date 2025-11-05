from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import math

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Tamaño de lote para procesar títulos (ajusta según memoria/CPU)
BATCH_SIZE = 256
MODEL_NAME = 'all-MiniLM-L6-v2'


def embed_titles(batch_size=BATCH_SIZE):
    """Genera embeddings para el campo `titulo` y guarda en `embedding_titulo`.

    Flujo:
    - Lee nodos que tengan `titulo` (streaming desde Neo4j)
    - Procesa en lotes: codifica con SentenceTransformer y actualiza en Neo4j
    - Usa `id(n)` para referenciar y actualizar el mismo nodo sin perder otras propiedades
    """

    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Cargar modelo de embeddings una sola vez (costoso)
    print(f"🧠 Cargando modelo de embeddings '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    total_processed = 0
    total_updated = 0

    try:
        with driver.session() as session:
            # Ejecutar query que devuelve id(n) y titulo para todos los nodos que tienen titulo
            # CAMBIO: Reemplazado exists(n.titulo) por n.titulo IS NOT NULL (sintaxis Neo4j moderna)
            # Usamos un cursor iterable para no traer todo a memoria de golpe
            print("📊 Obteniendo nodos con 'titulo' (streaming)...")
            result = session.run(
                "MATCH (n) WHERE n.titulo IS NOT NULL RETURN id(n) AS nid, n.titulo AS titulo"
            )

            batch = []  # lista de tuples (nid, titulo)

            for record in result:
                nid = record["nid"]
                titulo = record["titulo"]

                # MEJORA: Validación más robusta de títulos nulos/vacíos
                # Saltar títulos nulos o vacíos completamente
                if titulo is None or titulo == "":
                    continue
                
                # Normalizar a string y limpiar espacios en blanco
                titulo_text = str(titulo).strip()
                
                # Saltar si queda vacío después de limpiar o es muy corto
                if titulo_text == "" or len(titulo_text) < 3:
                    continue

                batch.append((nid, titulo_text))

                # Si alcanzamos el tamaño de lote, procesamos
                if len(batch) >= batch_size:
                    updated = _process_and_update_batch(session, model, batch)
                    total_processed += len(batch)
                    total_updated += updated
                    print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated})")
                    batch = []

            # Procesar remanente
            if batch:
                updated = _process_and_update_batch(session, model, batch)
                total_processed += len(batch)
                total_updated += updated
                print(f"   ✅ Procesadas {total_processed} (actualizadas: {total_updated})")

    except Exception as e:
        print(f"❌ Error general durante el proceso: {e}")
        raise

    finally:
        driver.close()
        print(f"\n🔌 Conexión cerrada. Total procesado: {total_processed}. Total actualizado: {total_updated}.")


def _process_and_update_batch(session, model, batch):
    """Codifica una lista de (nid, titulo) y actualiza los nodos en una sola transacción.

    Devuelve la cantidad de nodos actualizados exitosamente.
    """
    nids = [item[0] for item in batch]
    titles = [item[1] for item in batch]

    # Generar embeddings en lote (devuelve numpy array)
    embeddings = model.encode(titles, show_progress_bar=False)

    # Convertir embeddings a listas nativas de Python para Neo4j
    rows = []
    for nid, emb in zip(nids, embeddings):
        # emb puede ser numpy array; convertir a lista de floats
        emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))
        rows.append({"id": int(nid), "embedding": emb_list})

    # Actualizar en una sola query usando UNWIND
    update_query = (
        "UNWIND $rows AS row\n"
        "MATCH (n) WHERE id(n) = row.id\n"
        "SET n.embedding_titulo = row.embedding\n"
        "RETURN count(n) AS updated"
    )

    try:
        tx_result = session.run(update_query, rows=rows)
        rec = tx_result.single()
        updated = rec["updated"] if rec else 0
        return updated
    except Exception as e:
        print(f"❌ Error actualizando lote: {e}")
        # Como fallback intentar actualizar individualmente para encontrar nodos problemáticos
        updated = 0
        for row in rows:
            try:
                session.run(
                    "MATCH (n) WHERE id(n) = $nid SET n.embedding_titulo = $embedding",
                    nid=row["id"], embedding=row["embedding"]
                )
                updated += 1
            except Exception as inner_e:
                print(f"   ⚠️ Error actualizando nodo {row['id']}: {inner_e}")
        return updated


if __name__ == '__main__':
    # Ajusta BATCH_SIZE si necesitas más/menos paralelismo o memoria
    embed_titles(batch_size=BATCH_SIZE)
