"""
db/client.py — Neo4j client, schema init, and graph operations
===============================================================

All direct Neo4j interactions live here. Tool handlers import from this
module — they never touch the driver or write Cypher directly.

Responsibilities:
    - Driver singleton (one connection pool for the server's lifetime)
    - Schema initialisation (constraints + vector index)
    - CRUD helpers: profile, category, fact nodes
    - Duplicate detection via vector index
    - Semantic search + graph traversal

Graph schema maintained here:
    (Profile {id, created_at})
         └──[:HAS_CATEGORY]──▶ (Category {name, profile_id, created_at})
                                     └──[:HAS_FACT]──▶ (Fact {
                                                          id, text, embedding,
                                                          profile_id, category,
                                                          created_at
                                                        })
                                                            └──[:RELATED_TO]──▶ (Fact)
"""

import uuid
from datetime import datetime, timezone

from neo4j import GraphDatabase
from knowledge_graph_mcp.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBEDDING_DIMENSIONS, DUPLICATE_THRESHOLD, MAX_RELATED_FACTS,
    RELATED_TO_THRESHOLD, PROFILE_MATCH_THRESHOLD,
)

# ── Driver singleton ──────────────────────────────────────────────────────────
_driver = None


def get_driver():
    """
    Return the module-level Neo4j driver, creating it on first call.

    The driver manages a connection pool over Bolt. Creating it once at server
    startup (via initialise()) means every tool call reuses the pool — no
    per-request reconnection overhead.

    Raises:
        Exception: propagated to server.py startup handler if Neo4j is down.
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
    return _driver


# ── Schema initialisation ─────────────────────────────────────────────────────

def initialise():
    """
    Bootstrap Neo4j constraints and vector index. Called once at server startup.

    All statements use IF NOT EXISTS — fully idempotent, safe to re-run.

    Constraints created:
        - Profile.id UNIQUE   → prevents duplicate profiles
        - Fact.id UNIQUE      → UUID integrity for fact nodes

    Vector index created:
        - fact_embeddings on Fact.embedding (384 dims, cosine similarity)
        - Powers db.index.vector.queryNodes() for ANN semantic search
        - Without this index, vector search = full O(n) scan (unusable at scale)
    """
    driver = get_driver()
    with driver.session() as s:
        s.run("""
            CREATE CONSTRAINT profile_id_unique IF NOT EXISTS
            FOR (p:Profile) REQUIRE p.id IS UNIQUE
        """)
        s.run("""
            CREATE CONSTRAINT fact_id_unique IF NOT EXISTS
            FOR (f:Fact) REQUIRE f.id IS UNIQUE
        """)
        s.run(f"""
            CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
            FOR (f:Fact) ON (f.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIMENSIONS},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)


# ── Profile helpers ───────────────────────────────────────────────────────────

def upsert_profile(profile_id: str) -> None:
    """MERGE a Profile node — creates it on first use, no-ops after."""
    with get_driver().session() as s:
        s.run("""
            MERGE (p:Profile {id: $id})
            ON CREATE SET p.created_at = $ts
        """, id=profile_id, ts=_now())


def upsert_category(profile_id: str, category: str) -> None:
    """MERGE a Category node and link it to its Profile."""
    with get_driver().session() as s:
        s.run("""
            MATCH (p:Profile {id: $pid})
            MERGE (c:Category {name: $cat, profile_id: $pid})
            ON CREATE SET c.created_at = $ts
            MERGE (p)-[:HAS_CATEGORY]->(c)
        """, pid=profile_id, cat=category, ts=_now())


def list_profiles() -> list[dict]:
    """
    Return all profiles with fact counts, sorted by fact count descending.

    Traverses Profile → Category → Fact to count. OPTIONAL MATCH so
    empty profiles (no facts yet) are still returned with count 0.
    """
    with get_driver().session() as s:
        records = s.run("""
            MATCH (p:Profile)
            OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(:Category)-[:HAS_FACT]->(f:Fact)
            RETURN p.id AS id, p.created_at AS created_at, COUNT(f) AS fact_count
            ORDER BY fact_count DESC
        """)
        return [r.data() for r in records if r["id"] is not None]


def list_categories(profile_id: str) -> list[dict]:
    """Return all categories in a profile with fact counts, sorted descending."""
    with get_driver().session() as s:
        records = s.run("""
            MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category)
            OPTIONAL MATCH (c)-[:HAS_FACT]->(f:Fact)
            RETURN c.name AS name, COUNT(f) AS fact_count
            ORDER BY fact_count DESC
        """, pid=profile_id)
        return [r.data() for r in records]


# ── Duplicate detection ───────────────────────────────────────────────────────

def find_duplicate(embedding: list[float], profile_id: str) -> str | None:
    """
    Return the text of an existing fact if it is semantically identical
    to the incoming embedding (cosine similarity > DUPLICATE_THRESHOLD).

    Uses the vector index for ANN search — fast regardless of graph size.
    Profile isolation is enforced at index scan time via WHERE clause.

    Returns:
        str  → text of the duplicate fact
        None → no duplicate found, safe to store
    """
    with get_driver().session() as s:
        result = s.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', 1, $emb)
            YIELD node, score
            WHERE node.profile_id = $pid AND score > $threshold
            RETURN node.text AS text
            LIMIT 1
        """, emb=embedding, pid=profile_id, threshold=DUPLICATE_THRESHOLD)
        record = result.single()
        return record["text"] if record else None


# ── Fact write ────────────────────────────────────────────────────────────────

def write_fact(text: str, embedding: list[float], profile_id: str, category: str) -> tuple[str, int]:
    """
    Create a Fact node, link it to its Category, then wire up [:RELATED_TO]
    edges to any existing facts in the same profile+category that are
    semantically similar (cosine similarity > RELATED_TO_THRESHOLD).

    Returns:
        tuple[str, int]: (UUID of the new fact, count of RELATED_TO edges created)
    """
    fact_id = str(uuid.uuid4())
    with get_driver().session() as s:
        s.run("""
            MATCH (c:Category {name: $cat, profile_id: $pid})
            CREATE (f:Fact {
                id:         $fid,
                text:       $text,
                embedding:  $emb,
                profile_id: $pid,
                category:   $cat,
                created_at: $ts
            })
            CREATE (c)-[:HAS_FACT]->(f)
        """,
            fid=fact_id, text=text, emb=embedding,
            pid=profile_id, cat=category, ts=_now(),
        )

    related_count = _link_related_facts(fact_id, embedding, profile_id, category)
    return fact_id, related_count


def _link_related_facts(fact_id: str, embedding: list[float], profile_id: str, category: str) -> int:
    """
    After a new Fact node is written, find existing facts in the same
    profile+category whose embedding is within RELATED_TO_THRESHOLD
    and create bidirectional [:RELATED_TO] edges.

    Why bidirectional:
        Querying goes (matched_fact)-[:RELATED_TO]->(neighbour).
        Without a reverse edge, the neighbour can't find the new fact
        when the neighbour itself is the ANN match in a future query.

    Why same category filter:
        Facts across categories are related via the Profile/Category
        structure already. [:RELATED_TO] is for tighter semantic
        clustering within a category (e.g. all JWT-related auth facts
        connected to each other).

    Returns:
        int: Number of RELATED_TO edges created (each pair = 2 edges).
    """
    with get_driver().session() as s:
        # ANN search — up to 20 candidates, filtered to same profile+category,
        # excluding the new fact itself, above the relatedness threshold.
        related = s.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', 20, $emb)
            YIELD node, score
            WHERE node.profile_id = $pid
              AND node.category   = $cat
              AND node.id        <> $fid
              AND score           > $threshold
            RETURN node.id AS id
        """, emb=embedding, pid=profile_id, cat=category,
             fid=fact_id, threshold=RELATED_TO_THRESHOLD).data()

        count = 0
        for r in related:
            # MERGE prevents duplicate edges if this is called more than once
            s.run("""
                MATCH (f1:Fact {id: $fid1}), (f2:Fact {id: $fid2})
                MERGE (f1)-[:RELATED_TO]->(f2)
                MERGE (f2)-[:RELATED_TO]->(f1)
            """, fid1=fact_id, fid2=r["id"])
            count += 1

    return count


def find_best_profile(embedding: list[float]) -> str | None:
    """
    Search across ALL profiles for the closest matching fact to the given
    embedding. If the top match exceeds PROFILE_MATCH_THRESHOLD, return
    its profile_id so the caller can assign the new fact there.

    Used by store_fact when profile_id is not explicitly provided.

    Why top-1 across all profiles:
        The single closest fact tells us which project/namespace this
        fact most naturally belongs to. We don't need a per-profile
        average — one strong signal is enough.

    Returns:
        str  → profile_id of the best matching existing profile
        None → no profile matched well enough; caller should create a new one
    """
    with get_driver().session() as s:
        result = s.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', 1, $emb)
            YIELD node, score
            WHERE score > $threshold
            RETURN node.profile_id AS profile_id, score
            ORDER BY score DESC
            LIMIT 1
        """, emb=embedding, threshold=PROFILE_MATCH_THRESHOLD).single()
        return result["profile_id"] if result else None


# ── Semantic search + graph traversal ────────────────────────────────────────

def semantic_search(query_embedding: list[float], profile_id: str, top_k: int) -> list[dict]:
    """
    Two-phase retrieval: vector similarity search → graph traversal.

    Phase 1 — ANN vector search:
        db.index.vector.queryNodes scans the fact_embeddings index.
        WHERE node.profile_id = $pid enforces hard profile isolation at
        index scan time (not a post-filter — other profiles never scored).

    Phase 2 — Graph traversal:
        For each matched Fact, follow [:RELATED_TO] edges to fetch neighbours.
        OPTIONAL MATCH ensures unconnected facts are still returned.

    Returns:
        List of dicts with fact text, category, similarity score, related facts.
    """
    results = []
    with get_driver().session() as s:

        # Phase 1: vector similarity search within profile
        matched = s.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', $k, $emb)
            YIELD node, score
            WHERE node.profile_id = $pid
            RETURN node.id AS id, node.text AS text,
                   node.category AS category, node.created_at AS created_at,
                   score
            ORDER BY score DESC
        """, k=top_k, emb=query_embedding, pid=profile_id).data()

        # Phase 2: graph traversal from each matched node
        for fact in matched:
            related = s.run("""
                MATCH (f:Fact {id: $fid})
                OPTIONAL MATCH (f)-[:RELATED_TO]->(r:Fact)
                RETURN r.text AS text
                LIMIT $limit
            """, fid=fact["id"], limit=MAX_RELATED_FACTS).data()

            results.append({
                "fact":             fact["text"],
                "category":         fact["category"],
                "similarity_score": round(fact["score"], 4),
                "created_at":       fact["created_at"],
                "related_facts":    [r["text"] for r in related if r["text"]],
            })

    return results


# ── Internal helper ───────────────────────────────────────────────────────────

def _now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
