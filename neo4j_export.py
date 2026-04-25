# ─────────────────────────────────────────────────────
# neo4j_export.py — One-time Neo4j export for PPT screenshot
#
# SETUP (do this once):
#   1. Download Neo4j Desktop: https://neo4j.com/download/
#   2. Create a project, start a local database
#   3. Set the password below to match your database password
#   4. Run: python neo4j_export.py
#   5. Open http://localhost:7474 in your browser
#   6. Run this Cypher query to visualise:
#        MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150
#   7. Arrange the graph, take a full-resolution screenshot
#   8. Put the screenshot on your PPT slide (Knowledge Graph slide)
#   You do NOT need Neo4j running during the presentation.
# ─────────────────────────────────────────────────────

import pickle, sys
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[!] Install the Neo4j driver: pip install neo4j")
    sys.exit(1)

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"   # change to match your DB

DATA_DIR = Path(__file__).parent / "data"
with open(DATA_DIR / "graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"  Connecting to Neo4j at {NEO4J_URI} ...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph.")

    nodes_data = [
        {"id": nk,
         "label": d.get("label", nk),
         "etype": d.get("entity_type", "DEFAULT"),
         "btw": round(d.get("betweenness_centrality", 0.0), 4),
         "deg": round(d.get("degree_centrality", 0.0), 4)}
        for nk, d in G.nodes(data=True)
    ]
    session.run("""
        UNWIND $rows AS row
        CREATE (:Entity {
          id: row.id, label: row.label,
          entity_type: row.etype,
          betweenness: row.btw,
          degree: row.deg
        })
    """, rows=nodes_data)
    print(f"  Exported {len(nodes_data)} nodes.")

    edges_data = [
        {"src": s, "dst": t,
         "pred": d.get("predicate", "relates_to"),
         "w": d.get("weight", 1)}
        for s, t, d in G.edges(data=True)
        if s in G.nodes() and t in G.nodes()
    ]
    session.run("""
        UNWIND $rows AS row
        MATCH (a:Entity {id: row.src}), (b:Entity {id: row.dst})
        CREATE (a)-[:RELATES_TO {predicate: row.pred, weight: row.w}]->(b)
    """, rows=edges_data)
    print(f"  Exported {len(edges_data)} edges.")

driver.close()

print(f"""──────────────────────────────────────────────────────
  Export complete.
  Open: http://localhost:7474
  Login: neo4j / {NEO4J_PASSWORD}
  Run this query to see your full knowledge graph:

    MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150

  Arrange the layout → take screenshot → use in PPT.
──────────────────────────────────────────────────────""")
