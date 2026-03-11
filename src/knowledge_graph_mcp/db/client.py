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
    - Preview and atomic deletion with cascade cleanup

Graph hierarchy — Profile → Category → Fact:
    (Profile {id, created_at})
         └──[:HAS_CATEGORY]──▶ (Category {name, profile_id, created_at})
                                     └──[:HAS_FACT]──▶ (Fact {
                                                          id, text, embedding,
                                                          profile_id, category,
                                                          created_at
                                                        })
                                                            └──[:RELATED_TO]──▶ (Fact)

Key design decisions:
    - Profile   = project namespace (e.g. "octa", "personal"). Deleted when empty.
    - Category  = knowledge domain within a profile (e.g. "auth", "api"). Deleted when empty.
    - Fact      = single atomic statement. Linked to semantically similar facts via [:RELATED_TO].
    - Cascade   = deleting facts → empty category deleted → empty profile deleted.
    - Profile isolation is enforced everywhere: queries/deletions never cross profile boundaries.
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


def list_facts(profile_id: str, category: str) -> list[dict]:
    """
    Return all facts in a profile+category with id, text, created_at.
    Used by the delete workflow so users can browse fact IDs before targeting specific ones.
    Results ordered by creation time (oldest first = natural reading order).
    """
    with get_driver().session() as s:
        records = s.run("""
            MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category {name: $cat, profile_id: $pid})
            MATCH (c)-[:HAS_FACT]->(f:Fact)
            RETURN f.id AS id, f.text AS text, f.created_at AS created_at
            ORDER BY f.created_at ASC
        """, pid=profile_id, cat=category)
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


# ── Delete: shows preview before deletion (dry-run) ────────────────────────────────────────────────

def preview_delete_scope(
    scope: str,
    profile_id: str,
    category: str | None = None,
    fact_ids: list[str] | None = None,
) -> dict:
    """
    Dry-run: compute the full impact of a deletion without touching the graph.

    Scope options:
        "profile"  — all facts in all categories of the profile
        "category" — all facts inside one category
        "facts"    — specific facts by UUID list

    Returns:
        facts_in_scope  — list of {id, text, category} that will be deleted
        scoped_ids      — just the UUID set (used by delete_by_scope)
        cross_edges     — RELATED_TO edges that cross outside the deletion boundary;
                          these connect a scoped fact to a fact NOT in scope.
                          Used to warn the user about unlinked connections.
        cascade         — {categories: [...names...], profile: bool}
                          categories/profile that will be auto-cleaned because
                          they'll have 0 children after deletion.

    Profile isolation: all queries are scoped to profile_id — never touches other profiles.
    """
    with get_driver().session() as s:

        # ── Step 1: collect facts that are in scope ────────────────────────────
        # Profile → Category → Fact traversal scoped to the deletion target.
        if scope == "profile":
            scoped = s.run("""
                MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category)-[:HAS_FACT]->(f:Fact)
                RETURN f.id AS id, f.text AS text, c.name AS category
            """, pid=profile_id).data()

        elif scope == "category":
            scoped = s.run("""
                MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->
                      (c:Category {name: $cat, profile_id: $pid})-[:HAS_FACT]->(f:Fact)
                RETURN f.id AS id, f.text AS text, c.name AS category
            """, pid=profile_id, cat=category).data()

        elif scope == "facts":
            # Verify the given fact_ids belong to this profile (safety guard)
            scoped = s.run("""
                MATCH (f:Fact)
                WHERE f.id IN $ids AND f.profile_id = $pid
                RETURN f.id AS id, f.text AS text, f.category AS category
            """, ids=fact_ids or [], pid=profile_id).data()

        else:
            return {"error": f"Unknown scope '{scope}'. Use: profile | category | facts"}

        scoped_ids = {f["id"] for f in scoped}

        # ── Step 2: find cross-scope RELATED_TO edges ─────────────────────────
        # An edge "crosses the boundary" if one end is in scope and the other is not.
        # These will be unlinked when the scoped fact is deleted.
        # Only need one direction per pair — (scoped)→(outside) covers each unique pair.
        cross_edges = []
        if scoped_ids:
            cross_edges = s.run("""
                MATCH (f1:Fact)-[:RELATED_TO]->(f2:Fact)
                WHERE f1.id IN $ids AND NOT f2.id IN $ids
                RETURN f1.id        AS from_id,
                       f1.text      AS from_text,
                       f1.category  AS from_category,
                       f2.id        AS to_id,
                       f2.text      AS to_text,
                       f2.category  AS to_category
            """, ids=list(scoped_ids)).data()

        # ── Step 3: compute cascade — which nodes become empty after deletion ──
        # Cascade rule: empty Category → deleted; empty Profile → deleted.
        cascade_categories: list[str] = []
        cascade_profile = False

        if scope == "profile":
            # Entire profile goes — all its categories cascade too
            cascade_profile = True
            cats = s.run("""
                MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category)
                RETURN c.name AS name
            """, pid=profile_id).data()
            cascade_categories = [c["name"] for c in cats]

        elif scope == "category":
            # The targeted category is fully emptied (it IS the scope)
            cascade_categories = [category]
            # Profile cascades only if this was its last category
            other_count = s.run("""
                MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category)
                WHERE c.name <> $cat
                RETURN COUNT(c) AS count
            """, pid=profile_id, cat=category).single()["count"]
            cascade_profile = (other_count == 0)

        elif scope == "facts":
            # Check each affected category: will it have 0 facts after deletion?
            affected_cats = {f["category"] for f in scoped}
            for cat in affected_cats:
                total_in_cat = s.run("""
                    MATCH (c:Category {name: $cat, profile_id: $pid})-[:HAS_FACT]->(f:Fact)
                    RETURN COUNT(f) AS count
                """, cat=cat, pid=profile_id).single()["count"]
                deleting_from_cat = sum(1 for f in scoped if f["category"] == cat)
                if (total_in_cat - deleting_from_cat) == 0:
                    cascade_categories.append(cat)

            # Profile cascades if ALL its categories are being emptied
            if cascade_categories:
                total_cats = s.run("""
                    MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(c:Category)
                    RETURN COUNT(c) AS count
                """, pid=profile_id).single()["count"]
                cascade_profile = (len(cascade_categories) >= total_cats)

        return {
            "facts_in_scope": scoped,
            "scoped_ids":     list(scoped_ids),
            "cross_edges":    cross_edges,
            "cascade": {
                "categories": cascade_categories,
                "profile":    cascade_profile,
            },
        }


# ── Delete: atomic execution ──────────────────────────────────────────────────

def delete_by_scope(
    scope: str,
    profile_id: str,
    category: str | None = None,
    fact_ids: list[str] | None = None,
) -> dict:
    """
    Atomically delete facts + edges + cascade-clean empty categories/profiles.

    Deletion order is strict — Neo4j cannot delete a node that still has relationships:
        1. Delete RELATED_TO edges (in + out) on all scoped facts
        2. Delete HAS_FACT edges + Fact nodes
        3. Cascade: delete Category nodes that are now empty (no [:HAS_FACT] children)
        4. Cascade: delete Profile node if it has no [:HAS_CATEGORY] children left

    All four steps run inside a single write transaction — either all succeed or
    none do. No partial deletes left in the graph.

    Args:
        scope      — "profile" | "category" | "facts"
        profile_id — target profile (isolation boundary)
        category   — required when scope="category"
        fact_ids   — required when scope="facts"

    Returns:
        Dict with counts: facts_deleted, edges_removed, categories_deleted, profile_deleted.
    """
    with get_driver().session() as s:

        # ── Phase 1: read — resolve which fact IDs to delete ──────────────────
        # Done outside the write transaction (read-only, no retry concerns).
        if scope == "profile":
            rows = s.run("""
                MATCH (p:Profile {id: $pid})-[:HAS_CATEGORY]->(:Category)-[:HAS_FACT]->(f:Fact)
                RETURN f.id AS id
            """, pid=profile_id).data()
            target_ids = [r["id"] for r in rows]

        elif scope == "category":
            rows = s.run("""
                MATCH (:Category {name: $cat, profile_id: $pid})-[:HAS_FACT]->(f:Fact)
                RETURN f.id AS id
            """, cat=category, pid=profile_id).data()
            target_ids = [r["id"] for r in rows]

        elif scope == "facts":
            # Re-verify ownership — don't blindly trust caller's list
            rows = s.run("""
                MATCH (f:Fact)
                WHERE f.id IN $ids AND f.profile_id = $pid
                RETURN f.id AS id
            """, ids=fact_ids or [], pid=profile_id).data()
            target_ids = [r["id"] for r in rows]

        else:
            return {"error": f"Unknown scope '{scope}'"}

        if not target_ids:
            return {"status": "nothing_to_delete"}

        # ── Phase 2: write — all deletions in one atomic transaction ──────────
        def _delete_tx(tx):
            # Step 1: remove RELATED_TO edges (both directions) for all scoped facts.
            # The undirected `-[r]-` pattern matches edges where the fact is either end.
            edges = tx.run("""
                MATCH (f:Fact)-[r:RELATED_TO]-()
                WHERE f.id IN $ids
                DELETE r
                RETURN COUNT(r) AS count
            """, ids=target_ids).single()["count"]

            # Step 2: remove HAS_FACT edges and delete the Fact nodes themselves
            facts = tx.run("""
                MATCH (c:Category)-[r:HAS_FACT]->(f:Fact)
                WHERE f.id IN $ids
                DELETE r, f
                RETURN COUNT(f) AS count
            """, ids=target_ids).single()["count"]

            # Step 3: cascade — delete any Category under this profile that is now empty
            cats = tx.run("""
                MATCH (p:Profile {id: $pid})-[r:HAS_CATEGORY]->(c:Category)
                WHERE NOT (c)-[:HAS_FACT]->()
                DELETE r, c
                RETURN COUNT(c) AS count
            """, pid=profile_id).single()["count"]

            # Step 4: cascade — delete the Profile if it has no categories left
            prof = tx.run("""
                MATCH (p:Profile {id: $pid})
                WHERE NOT (p)-[:HAS_CATEGORY]->()
                DELETE p
                RETURN COUNT(p) AS count
            """, pid=profile_id).single()["count"]

            return edges, facts, cats, prof

        edges_del, facts_del, cats_del, prof_del = s.execute_write(_delete_tx)

        return {
            "status":              "deleted",
            "facts_deleted":       facts_del,
            "edges_removed":       edges_del,
            "categories_deleted":  cats_del,
            "profile_deleted":     prof_del > 0,
        }


# ── Internal helper ───────────────────────────────────────────────────────────

def _now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
