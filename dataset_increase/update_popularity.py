from neo4j import GraphDatabase

# Configuración de Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tu_password"  # CAMBIAR POR TU PASSWORD

def update_popularity_property():
    # Conectar a Neo4j
    print("Conectando a Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Actualizar la propiedad popularity basada en shares
            print("Actualizando propiedad 'popularity' en todos los nodos Noticia...")
            
            result = session.run("""
                MATCH (n:Noticia)
                SET n.popularity = CASE 
                    WHEN n.shares > 1400 THEN true 
                    ELSE false 
                END
                RETURN count(n) AS total_updated
            """)
            
            total_updated = result.single()["total_updated"]
            
            # Contar cuántos quedaron con popularity = True
            result_popular = session.run("""
                MATCH (n:Noticia)
                WHERE n.popularity = true
                RETURN count(n) AS popular_count
            """)
            
            popular_count = result_popular.single()["popular_count"]
            
            print(f"✅ Nodos actualizados en total: {total_updated}")
            print(f"✅ Nodos con popularity = True: {popular_count}")
            print(f"✅ Nodos con popularity = False: {total_updated - popular_count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        driver.close()
        print("Conexión cerrada.")

if __name__ == "__main__":
    update_popularity_property()