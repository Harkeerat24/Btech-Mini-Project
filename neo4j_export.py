# ─────────────────────────────────────────────────────
# neo4j_export.py — One-time Neo4j export for PPT screenshot
#
# SETUP (do this once):
#   1. In Neo4j Aura, create/start your DB instance
#   2. Set environment variables:
#        set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687
#        set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443
#        set NEO4J_USERNAME=9b183348
#        set NEO4J_DATABASE=9b183348
#        set NEO4J_PASSWORD=<your-password>
#   3. Run: python neo4j_export.py
#   4. Open Neo4j Browser from Aura Console
#   5. Run this Cypher query to visualise:
#        MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150
#   6. Arrange the graph, take a full-resolution screenshot
#   7. Put the screenshot on your PPT slide (Knowledge Graph slide)
#   You do NOT need Neo4j running during the presentation.
# ─────────────────────────────────────────────────────

import os
import pickle
import shutil
import sys
import textwrap
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
except ImportError:
    print("[!] Install the Neo4j driver: pip install neo4j")
    sys.exit(1)

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _neo4j_uri_candidates(uri: str) -> list[str]:
    """Prefer Bolt over Neo4j routing, with Aura port fallbacks."""
    if not uri:
        return []

    parsed = urlparse(uri)
    host = parsed.hostname
    if not host:
        return [uri]

    scheme = "bolt+s"
    ports: list[int] = []
    if parsed.port is not None:
        ports.append(parsed.port)
    for port in (7687, 443):
        if port not in ports:
            ports.append(port)

    return [f"{scheme}://{host}:{port}" for port in ports]


def _is_auth_error(error: Exception) -> bool:
    text = str(error)
    return "Unauthorized" in text or "AuthError" in text or "authentication failure" in text.lower()


uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
db = os.getenv("NEO4J_DATABASE")

if not uri:
    print("[!] Missing NEO4J_URI environment variable.")
    print("    Example: set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687")
    print("    Fallback: set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443")
    sys.exit(1)

if not password:
    print("[!] Missing NEO4J_PASSWORD environment variable.")
    print("    Example (Aura):")
    print("    set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:7687")
    print("    set NEO4J_URI=bolt+s://<instance-id>.databases.neo4j.io:443")
    print("    set NEO4J_USERNAME=9b183348")
    print("    set NEO4J_DATABASE=9b183348")
    print("    set NEO4J_PASSWORD=<your-password>")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
with open(DATA_DIR / "graph.pkl", "rb") as f:
    G = pickle.load(f)

TERM_WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 120)
SEP = "─" * TERM_WIDTH

print(f"  Connecting to Neo4j at {uri} ...")
print(f"  User: {user}  ·  Database: {db}")
driver = GraphDatabase.driver(uri, auth=(user, password))

try:
    driver.verify_connectivity()
    print("  Connection verified.")
except Exception as e:
    print(f"[!] Cannot reach Neo4j: {e}")
    print("    Check your URI and password in .env")
    sys.exit(1)

with driver.session(database=db) as session:
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

print(SEP)
print("  Export complete.")
for line in textwrap.wrap(
        "Open Neo4j Browser (Aura) from your Aura Console.",
        width=max(30, TERM_WIDTH - 4),
):
    print(f"  {line}")
print("  Run this query to see your full knowledge graph:")
print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150")
for line in textwrap.wrap(
        "Arrange the layout, take screenshot, and use it in PPT.",
        width=max(30, TERM_WIDTH - 4),
):
    print(f"  {line}")
print(SEP)
