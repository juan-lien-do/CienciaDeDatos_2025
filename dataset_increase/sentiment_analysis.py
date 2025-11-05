from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

# Configuración de modelos
SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
EMOTION_MODEL = 'cardiffnlp/twitter-roberta-base-emotion'

# Configuración optimizada para GTX 1660 Super (6GB VRAM)
BATCH_SIZE_GPU = 128  # Tamaño optimizado para GPU
BATCH_SIZE_CPU = 32   # Tamaño conservador para CPU


def analyze_sentiment_and_emotions():
    """
    Realiza análisis de sentimientos y emociones en los títulos de noticias.
    
    Flujo:
    1. Conecta a Neo4j y obtiene títulos sin análisis
    2. Carga modelos de Twitter-RoBERTa para sentimiento y emociones
    3. Procesa títulos en lotes para eficiencia
    4. Guarda resultados en analisis_sentimiento_titulo y analisis_emocion_titulo
    
    Modelos utilizados:
    - Sentimiento: twitter-roberta-base-sentiment-latest (POSITIVE, NEGATIVE, NEUTRAL)
    - Emociones: twitter-roberta-base-emotion (JOY, ANGER, FEAR, SADNESS, etc.)
    """
    
    print("🚀 Iniciando análisis de sentimientos y emociones con Twitter-RoBERTa")
    print("=" * 80)
    
    # Conectar a Neo4j
    print("🔗 Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Cargar modelos de análisis
        sentiment_pipeline, emotion_pipeline = load_analysis_models()
        
        # Procesar títulos
        with driver.session() as session:
            process_titles_analysis(session, sentiment_pipeline, emotion_pipeline)
        
        # Mostrar estadísticas finales
        show_analysis_statistics(driver)
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        driver.close()
        print("🔌 Conexión cerrada.")


def load_analysis_models():
    """
    Carga los modelos de análisis de sentimientos y emociones.
    
    Returns:
        tuple: (sentiment_pipeline, emotion_pipeline)
    """
    print("\n🤖 Cargando modelos de análisis...")
    
    # Verificar CUDA y configurar dispositivo
    if torch.cuda.is_available():
        device = 0  # Primera GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"   🎮 GPU detectada: {gpu_name}")
        print(f"   💾 Memoria GPU: {gpu_memory:.1f} GB")
        print(f"   ⚡ Usando CUDA para aceleración")
        
        # Limpiar memoria GPU previa
        torch.cuda.empty_cache()
        
    else:
        device = -1  # CPU fallback
        print(f"   💻 CUDA no disponible, usando CPU")
        print(f"   💡 Instala PyTorch con CUDA para usar tu GTX 1660 Super")
    
    try:
        # Cargar pipeline de sentimientos
        print(f"   😊 Cargando modelo de sentimientos: {SENTIMENT_MODEL}")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=device,
            return_all_scores=True
        )
        print("   ✅ Modelo de sentimientos cargado")
        
        # Cargar pipeline de emociones
        print(f"   😭 Cargando modelo de emociones: {EMOTION_MODEL}")
        emotion_pipeline = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            tokenizer=EMOTION_MODEL,
            device=device,
            return_all_scores=True
        )
        print("   ✅ Modelo de emociones cargado")
        
        return sentiment_pipeline, emotion_pipeline
        
    except Exception as e:
        print(f"   ❌ Error cargando modelos: {e}")
        print("   💡 Instalando modelos automáticamente...")
        
        # Reintentar con descarga automática
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device,
            return_all_scores=True
        )
        
        emotion_pipeline = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            device=device,
            return_all_scores=True
        )
        
        return sentiment_pipeline, emotion_pipeline


def process_titles_analysis(session, sentiment_pipeline, emotion_pipeline):
    """
    Procesa los títulos para análisis de sentimientos y emociones.
    
    Args:
        session: Sesión de Neo4j
        sentiment_pipeline: Pipeline de análisis de sentimientos
        emotion_pipeline: Pipeline de análisis de emociones
    """
    print("\n📊 Obteniendo títulos para análisis...")
    
    # Determinar tamaño de batch según dispositivo
    batch_size = BATCH_SIZE_GPU if torch.cuda.is_available() else BATCH_SIZE_CPU
    print(f"   🔧 Tamaño de batch: {batch_size} ({'GPU' if torch.cuda.is_available() else 'CPU'})")
    
            # Verificar títulos existentes sin análisis
    check_result = session.run("""
        MATCH (n) 
        WHERE n.titulo IS NOT NULL 
          AND (n.analisis_sentimiento_titulo_label IS NULL OR n.analisis_emocion_titulo_label IS NULL)
        RETURN count(n) AS pending_count
    """)
    pending_count = check_result.single()["pending_count"]
    print(f"   📈 Títulos pendientes de análisis: {pending_count}")
    
    if pending_count == 0:
        print("   ✅ Todos los títulos ya tienen análisis completo")
        return
    
    # Obtener títulos para procesar (TODOS los que faltan)
    result = session.run("""
        MATCH (n) 
        WHERE n.titulo IS NOT NULL 
          AND (n.analisis_sentimiento_titulo_label IS NULL OR n.analisis_emocion_titulo_label IS NULL)
        RETURN id(n) AS nid, n.titulo AS titulo
    """)
    
    # Procesar en lotes
    batch = []
    processed_count = 0
    
    for record in result:
        nid = record["nid"]
        titulo = record["titulo"]
        
        # Validar título
        if not titulo or len(str(titulo).strip()) < 3:
            continue
            
        batch.append((nid, str(titulo).strip()))
        
        # Procesar lote cuando esté lleno
        if len(batch) >= batch_size:
            process_batch_analysis(session, batch, sentiment_pipeline, emotion_pipeline)
            processed_count += len(batch)
            print(f"   ✅ Procesados {processed_count} títulos...")
            batch = []
    
    # Procesar último lote
    if batch:
        process_batch_analysis(session, batch, sentiment_pipeline, emotion_pipeline)
        processed_count += len(batch)
        print(f"   ✅ Procesados {processed_count} títulos...")
    
    print(f"\n🎉 Análisis completado para {processed_count} títulos")


def process_batch_analysis(session, batch, sentiment_pipeline, emotion_pipeline):
    """
    Procesa un lote de títulos para análisis de sentimientos y emociones.
    
    Args:
        session: Sesión de Neo4j
        batch: Lista de tuplas (nid, titulo)
        sentiment_pipeline: Pipeline de sentimientos
        emotion_pipeline: Pipeline de emociones
    """
    nids = [item[0] for item in batch]
    titles = [item[1] for item in batch]
    
    # Monitoreo de memoria GPU antes del procesamiento
    if torch.cuda.is_available():
        gpu_memory_before = torch.cuda.memory_allocated(0) / (1024**2)  # MB
    
    try:
        # Análisis de sentimientos
        print(f"      😊 Analizando sentimientos para {len(titles)} títulos...")
        sentiment_results = sentiment_pipeline(titles)
        
        # Análisis de emociones  
        print(f"      😭 Analizando emociones para {len(titles)} títulos...")
        emotion_results = emotion_pipeline(titles)
        
        # Procesar resultados y actualizar Neo4j
        update_batch_results(session, nids, titles, sentiment_results, emotion_results)
        
        # Monitoreo de memoria GPU después del procesamiento
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            memory_used = gpu_memory_after - gpu_memory_before
            if memory_used > 100:  # Si usa más de 100MB, mostrar info
                print(f"      🎮 Memoria GPU usada: {memory_used:.1f}MB")
            
            # Limpiar cache si la memoria está alta
            if gpu_memory_after > 4000:  # Más de 4GB en uso
                torch.cuda.empty_cache()
                print(f"      🧹 Limpiando cache GPU...")
        
    except Exception as e:
        print(f"      ❌ Error procesando lote: {e}")
        # Fallback: procesar individualmente
        process_individually(session, batch, sentiment_pipeline, emotion_pipeline)


def process_individually(session, batch, sentiment_pipeline, emotion_pipeline):
    """
    Procesa títulos individualmente como fallback.
    
    Args:
        session: Sesión de Neo4j
        batch: Lista de tuplas (nid, titulo)
        sentiment_pipeline: Pipeline de sentimientos
        emotion_pipeline: Pipeline de emociones
    """
    print("      🔄 Procesando individualmente como fallback...")
    
    for nid, titulo in batch:
        try:
            # Análisis individual
            sentiment_result = sentiment_pipeline([titulo])
            emotion_result = emotion_pipeline([titulo])
            
            # Actualizar individual
            update_batch_results(session, [nid], [titulo], sentiment_result, emotion_result)
            
        except Exception as e:
            print(f"      ⚠️ Error procesando título '{titulo[:50]}...': {e}")


def update_batch_results(session, nids, titles, sentiment_results, emotion_results):
    """
    Actualiza los resultados de análisis en Neo4j.
    
    Args:
        session: Sesión de Neo4j
        nids: Lista de IDs de nodos
        titles: Lista de títulos
        sentiment_results: Resultados de análisis de sentimientos
        emotion_results: Resultados de análisis de emociones
    """
    # Preparar datos para actualización
    update_data = []
    
    for i, nid in enumerate(nids):
        try:
            # Procesar resultado de sentimiento
            sentiment_scores = sentiment_results[i]
            best_sentiment = max(sentiment_scores, key=lambda x: x['score'])
            
            # Procesar resultado de emoción
            emotion_scores = emotion_results[i]
            best_emotion = max(emotion_scores, key=lambda x: x['score'])
            
            # Crear propiedades separadas compatible con Neo4j (solo primitivos)
            update_data.append({
                'id': int(nid),
                # Propiedades de sentimiento separadas
                'sentiment_label': best_sentiment['label'],
                'sentiment_score': float(best_sentiment['score']),
                'sentiment_all_labels': [s['label'] for s in sentiment_scores],
                'sentiment_all_scores': [float(s['score']) for s in sentiment_scores],
                # Propiedades de emoción separadas
                'emotion_label': best_emotion['label'], 
                'emotion_score': float(best_emotion['score']),
                'emotion_all_labels': [e['label'] for e in emotion_scores],
                'emotion_all_scores': [float(e['score']) for e in emotion_scores]
            })
            
        except Exception as e:
            print(f"      ⚠️ Error procesando resultado para nodo {nid}: {e}")
    
    # Actualizar en Neo4j usando UNWIND
    if update_data:
        try:
            update_query = """
                UNWIND $data AS row
                MATCH (n) WHERE id(n) = row.id
                SET n.analisis_sentimiento_titulo_label = row.sentiment_label,
                    n.analisis_sentimiento_titulo_score = row.sentiment_score,
                    n.analisis_sentimiento_titulo_all_labels = row.sentiment_all_labels,
                    n.analisis_sentimiento_titulo_all_scores = row.sentiment_all_scores,
                    n.analisis_emocion_titulo_label = row.emotion_label,
                    n.analisis_emocion_titulo_score = row.emotion_score,
                    n.analisis_emocion_titulo_all_labels = row.emotion_all_labels,
                    n.analisis_emocion_titulo_all_scores = row.emotion_all_scores
                RETURN count(n) AS updated
            """
            
            result = session.run(update_query, data=update_data)
            updated_count = result.single()["updated"]
            
        except Exception as e:
            print(f"      ❌ Error actualizando lote en Neo4j: {e}")
            # Fallback: actualizar uno por uno
            updated_count = 0
            for item in update_data:
                try:
                    session.run("""
                        MATCH (n) WHERE id(n) = $id
                        SET n.analisis_sentimiento_titulo_label = $sentiment_label,
                            n.analisis_sentimiento_titulo_score = $sentiment_score,
                            n.analisis_sentimiento_titulo_all_labels = $sentiment_all_labels,
                            n.analisis_sentimiento_titulo_all_scores = $sentiment_all_scores,
                            n.analisis_emocion_titulo_label = $emotion_label,
                            n.analisis_emocion_titulo_score = $emotion_score,
                            n.analisis_emocion_titulo_all_labels = $emotion_all_labels,
                            n.analisis_emocion_titulo_all_scores = $emotion_all_scores
                    """, 
                    id=item['id'], 
                    sentiment_label=item['sentiment_label'],
                    sentiment_score=item['sentiment_score'],
                    sentiment_all_labels=item['sentiment_all_labels'],
                    sentiment_all_scores=item['sentiment_all_scores'],
                    emotion_label=item['emotion_label'],
                    emotion_score=item['emotion_score'],
                    emotion_all_labels=item['emotion_all_labels'],
                    emotion_all_scores=item['emotion_all_scores'])
                    updated_count += 1
                except Exception as inner_e:
                    print(f"        ⚠️ Error actualizando nodo {item['id']}: {inner_e}")


def show_analysis_statistics(driver):
    """
    Muestra estadísticas finales del análisis realizado.
    
    Args:
        driver: Driver de Neo4j
    """
    print("\n📊 ESTADÍSTICAS FINALES DEL ANÁLISIS")
    print("=" * 60)
    
    try:
        with driver.session() as session:
            # Contar análisis completados
            stats_result = session.run("""
                MATCH (n) 
                RETURN 
                    count(CASE WHEN n.analisis_sentimiento_titulo_label IS NOT NULL THEN 1 END) AS sentiment_count,
                    count(CASE WHEN n.analisis_emocion_titulo_label IS NOT NULL THEN 1 END) AS emotion_count,
                    count(CASE WHEN n.titulo IS NOT NULL THEN 1 END) AS total_titles
            """)
            
            stats = stats_result.single()
            sentiment_count = stats["sentiment_count"]
            emotion_count = stats["emotion_count"]
            total_titles = stats["total_titles"]
            
            print(f"📈 Títulos con análisis de sentimiento: {sentiment_count}/{total_titles}")
            print(f"😊 Títulos con análisis de emoción: {emotion_count}/{total_titles}")
            
            # Distribución de sentimientos
            print(f"\n📊 DISTRIBUCIÓN DE SENTIMIENTOS:")
            sentiment_dist = session.run("""
                MATCH (n) 
                WHERE n.analisis_sentimiento_titulo_label IS NOT NULL
                RETURN n.analisis_sentimiento_titulo_label AS sentiment, 
                       count(*) AS count
                ORDER BY count DESC
            """)
            
            for record in sentiment_dist:
                sentiment = record["sentiment"]
                count = record["count"]
                percentage = (count / sentiment_count) * 100 if sentiment_count > 0 else 0
                print(f"   {sentiment}: {count} ({percentage:.1f}%)")
            
            # Top 5 emociones
            print(f"\n😭 TOP 5 EMOCIONES:")
            emotion_dist = session.run("""
                MATCH (n) 
                WHERE n.analisis_emocion_titulo_label IS NOT NULL
                RETURN n.analisis_emocion_titulo_label AS emotion, 
                       count(*) AS count
                ORDER BY count DESC
                LIMIT 5
            """)
            
            for record in emotion_dist:
                emotion = record["emotion"]
                count = record["count"]
                percentage = (count / emotion_count) * 100 if emotion_count > 0 else 0
                print(f"   {emotion}: {count} ({percentage:.1f}%)")
            
            # Ejemplos por sentimiento
            print(f"\n📝 EJEMPLOS POR SENTIMIENTO:")
            examples = session.run("""
                MATCH (n) 
                WHERE n.analisis_sentimiento_titulo_label IS NOT NULL
                WITH n.analisis_sentimiento_titulo_label AS sentiment, 
                     collect({titulo: n.titulo, score: n.analisis_sentimiento_titulo_score}) AS examples
                RETURN sentiment, examples[0..2] AS sample_examples
            """)
            
            for record in examples:
                sentiment = record["sentiment"]
                sample_examples = record["sample_examples"]
                print(f"\n   {sentiment}:")
                for example in sample_examples:
                    titulo = example['titulo'][:60] + "..." if len(example['titulo']) > 60 else example['titulo']
                    score = example['score']
                    print(f"     • {titulo} (score: {score:.3f})")
            
            # Mostrar estructura de datos guardada
            print(f"\n💾 ESTRUCTURA DE DATOS GUARDADA:")
            print("   📊 Propiedades de Sentimiento:")
            print("      - analisis_sentimiento_titulo_label: etiqueta principal")
            print("      - analisis_sentimiento_titulo_score: confianza principal")  
            print("      - analisis_sentimiento_titulo_all_labels: [todas las etiquetas]")
            print("      - analisis_sentimiento_titulo_all_scores: [todas las puntuaciones]")
            print("   😊 Propiedades de Emoción:")
            print("      - analisis_emocion_titulo_label: etiqueta principal")
            print("      - analisis_emocion_titulo_score: confianza principal")
            print("      - analisis_emocion_titulo_all_labels: [todas las etiquetas]")
            print("      - analisis_emocion_titulo_all_scores: [todas las puntuaciones]")
                
    except Exception as e:
        print(f"❌ Error mostrando estadísticas: {e}")


if __name__ == '__main__':
    print("🚀 Iniciando análisis de sentimientos y emociones con Twitter-RoBERTa")
    print("=" * 80)
    print("🎯 Objetivo: Analizar sentimientos y emociones en títulos de noticias")
    print("😊 Modelo sentimientos: twitter-roberta-base-sentiment-latest")
    print("😭 Modelo emociones: twitter-roberta-base-emotion")
    print("💾 Campos destino: analisis_sentimiento_titulo, analisis_emocion_titulo")
    print("🔬 Análisis: Etiquetas + scores + distribuciones completas")
    
    # Mostrar información de GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)} - OPTIMIZADO para tu GTX 1660 Super!")
    else:
        print("💡 Para usar tu GTX 1660 Super, instala PyTorch con CUDA:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 80)
    
    analyze_sentiment_and_emotions()
    
    print("\n" + "=" * 80)
    print("✅ Análisis de sentimientos y emociones completado")
    print("🔍 Los resultados están disponibles en propiedades separadas:")
    print("   📊 Sentimientos: analisis_sentimiento_titulo_label, analisis_sentimiento_titulo_score")
    print("   😊 Emociones: analisis_emocion_titulo_label, analisis_emocion_titulo_score")  
    print("   📈 Arrays completos: *_all_labels, *_all_scores")
    print("� Tip: Usa estos análisis para mejorar la predicción de viralidad")
    print("=" * 80)