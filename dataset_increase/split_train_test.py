from neo4j import GraphDatabase
import random

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def split_train_test():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Contar el total de nodos Noticia
            result = session.run("MATCH (n:Noticia) RETURN count(n) AS total")
            total_nodos = result.single()["total"]
            print(f"Total de nodos Noticia: {total_nodos}")
            
            if total_nodos == 0:
                print("No hay nodos Noticia para procesar.")
                return
            
            # Calcular cantidades para train/test
            test_size = int(total_nodos * 0.2)
            train_size = total_nodos - test_size
            
            print(f"Nodos para test (20%): {test_size}")
            print(f"Nodos para train (80%): {train_size}")
            
            # Obtener todas las URLs de los nodos
            result = session.run("MATCH (n:Noticia) RETURN n.url AS url")
            todas_urls = [record["url"] for record in result]
            
            # Seleccionar aleatoriamente URLs para test
            random.seed(42)  # Seed fijo para reproducibilidad
            urls_test = random.sample(todas_urls, test_size)
            
            print("Asignando subset = 'test' a nodos seleccionados...")
            # Actualizar nodos seleccionados como 'test'
            for url in urls_test:
                session.run("""
                    MATCH (n:Noticia {url: $url})
                    SET n.subset = 'test'
                """, url=url)
            
            print("Asignando subset = 'train' al resto...")
            # Actualizar el resto como 'train'
            session.run("""
                MATCH (n:Noticia)
                WHERE n.subset IS NULL OR n.subset <> 'test'
                SET n.subset = 'train'
            """)
            
            # Verificar resultados
            result_train = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'train'
                RETURN count(n) AS train_count
            """)
            
            result_test = session.run("""
                MATCH (n:Noticia)
                WHERE n.subset = 'test'
                RETURN count(n) AS test_count
            """)
            
            train_count = result_train.single()["train_count"]
            test_count = result_test.single()["test_count"]
            
            print(f"\n✅ División completada:")
            print(f"   Nodos con subset = 'train': {train_count}")
            print(f"   Nodos con subset = 'test': {test_count}")
            print(f"   Total verificado: {train_count + test_count}")
            
            # Verificar porcentajes
            print(f"\n📊 Porcentajes:")
            print(f"   Train: {train_count/total_nodos*100:.1f}%")
            print(f"   Test: {test_count/total_nodos*100:.1f}%")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        driver.close()
        print("\nConexión cerrada.")

if __name__ == "__main__":
    split_train_test()