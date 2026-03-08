"""
tools/query.py — query_knowledge tool
========================================

Retrieves semantically relevant facts from Neo4j using two-phase retrieval:
  Phase 1 — Vector similarity search (semantic matching via ANN index)
  Phase 2 — Graph traversal ([:RELATED_TO] neighbour expansion)

This combination is what distinguishes graph + vector from a pure vector DB:
  Pure vector DB → returns similar text chunks (flat, no relationships)
  This tool      → returns matched facts + their graph-connected neighbours
                   all strictly within the requested profile namespace
"""

import asyncio
from mcp.types import Tool
from knowledge_graph_mcp.db import client, embeddings

# ── Tool definition ───────────────────────────────────────────────────────────

DEFINITION = Tool(
    name="query_knowledge",
    description=(
        "Retrieve semantically relevant facts from the knowledge graph using "
        "natural language. Searches strictly within the specified profile — "
        "no cross-profile results ever returned. "
        "Semantic search means exact keyword matches are NOT required: "
        "'how does token auth work' will match 'JWT expiry is 15 minutes'. "
        "Also returns graph-connected related facts for richer context. "
        "Use list_profiles to confirm the profile exists before querying."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language question or topic. Use the user's phrasing "
                    "directly — the semantic search handles vocabulary mismatches. "
                    "Example: 'how does authentication handle token expiry'"
                ),
            },
            "profile_id": {
                "type": "string",
                "description": (
                    "Profile to search within. Results are hard-scoped to this "
                    "profile — other profiles are never touched."
                ),
            },
            "top_k": {
                "type":        "integer",
                "description": "Max results to return. Default 5. Use 8–10 for broad topics.",
                "default":     5,
                "minimum":     1,
                "maximum":     20,
            },
        },
        "required": ["query", "profile_id"],
    },
)


# ── Tool handler ──────────────────────────────────────────────────────────────

async def handle(args: dict) -> dict:
    """
    Execute the query_knowledge tool.

    Flow:
        1. Validate inputs
        2. Embed query text (thread pool — CPU-bound)
        3. ANN vector search within profile via Neo4j vector index
        4. Graph traversal from each matched node for related facts
        5. Return ranked results with similarity scores

    Args:
        args: Dict from Claude containing query, profile_id, top_k.

    Returns:
        Dict with status 'ok' and ranked results list, or 'no_results'.
    """
    query      = args.get("query", "").strip()
    profile_id = args.get("profile_id", "").strip()
    top_k      = int(args.get("top_k", 5))

    if not query:
        return {"status": "error", "message": "'query' is required and cannot be empty."}
    if not profile_id:
        return {"status": "error", "message": "'profile_id' is required and cannot be empty."}

    # Embed query in thread pool (same model + same vector space as stored facts)
    loop            = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, embeddings.embed_text, query)

    # Two-phase retrieval: vector similarity + graph traversal
    results = client.semantic_search(query_embedding, profile_id, top_k)

    if not results:
        return {
            "status":  "no_results",
            "profile": profile_id,
            "query":   query,
            "message": f"No facts found in '{profile_id}' matching this query.",
        }

    return {
        "status":        "ok",
        "profile":       profile_id,
        "query":         query,
        "results_count": len(results),
        "results":       results,
    }
