import os
import pickle
import shutil
import sys
import textwrap
from pathlib import Path

try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
except ImportError:
    print("[!] Install the Neo4j driver: pip install neo4j")
    sys.exit(1)

load_dotenv()


uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
db = os.getenv("NEO4J_DATABASE")

if not uri:
    print("[!] Missing NEO4J_URI environment variable.")
    print("    Example: NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687")
    sys.exit(1)

if not password:
    print("[!] Missing NEO4J_PASSWORD environment variable.")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
with open(DATA_DIR / "graph.pkl", "rb") as f:
    G = pickle.load(f)

TERM_WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 120)
SEP = "-" * TERM_WIDTH

print(f"  Connecting to Neo4j at {uri} ...")
print(f"  User: {user}  Database: {db}")
driver = GraphDatabase.driver(uri, auth=(user, password))

try:
    driver.verify_connectivity()
    print("  Connection verified.")
except Exception as exc:
    print(f"[!] Cannot reach Neo4j: {exc}")
    print("    Check your URI, database, username, and password in .env")
    sys.exit(1)

with driver.session(database=db) as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph.")

    nodes_data = [
        {
            "id": node_key,
            "label": data.get("label", node_key),
            "etype": data.get("entity_type", "DEFAULT"),
            "btw": round(data.get("betweenness_centrality", 0.0), 4),
            "deg": round(data.get("degree_centrality", 0.0), 4),
        }
        for node_key, data in G.nodes(data=True)
    ]
    session.run(
        """
        UNWIND $rows AS row
        CREATE (:Entity {
          id: row.id,
          label: row.label,
          entity_type: row.etype,
          betweenness: row.btw,
          degree: row.deg
        })
        """,
        rows=nodes_data,
    )
    print(f"  Exported {len(nodes_data)} nodes.")

    edges_data = [
        {
            "src": source,
            "dst": target,
            "pred": data.get("predicate", "relates_to"),
            "w": data.get("weight", 1),
        }
        for source, target, data in G.edges(data=True)
        if source in G.nodes() and target in G.nodes()
    ]
    session.run(
        """
        UNWIND $rows AS row
        MATCH (a:Entity {id: row.src}), (b:Entity {id: row.dst})
        CREATE (a)-[:RELATES_TO {predicate: row.pred, weight: row.w}]->(b)
        """,
        rows=edges_data,
    )
    print(f"  Exported {len(edges_data)} edges.")

driver.close()

print(SEP)
print("  Export complete.")
for line in textwrap.wrap(
    "Open Neo4j Browser from Aura Console to explore the knowledge graph.",
    width=max(30, TERM_WIDTH - 4),
):
    print(f"  {line}")
print("  Run this query to inspect the graph:")
print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150")
print(SEP)
