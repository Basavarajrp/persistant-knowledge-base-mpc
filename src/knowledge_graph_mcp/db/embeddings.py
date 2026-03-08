"""
db/embeddings.py — Embedding model singleton
=============================================

Owns the SentenceTransformer instance. Exposes a single embed_text() function
used by both the store and query tool handlers.

Singleton pattern: the model is loaded once on first call and reused for every
subsequent request — avoiding a ~3-second model reload on every tool call.
"""

from sentence_transformers import SentenceTransformer
from knowledge_graph_mcp.config import EMBEDDING_MODEL

# Module-level singleton — None until first call to embed_text()
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    First call: downloads model if not cached (~80MB to ~/.cache/huggingface/),
                then loads into memory (~500ms).
    All subsequent calls: returns the already-loaded model instantly.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Convert text into a 384-dimensional vector embedding.

    Uses the same model for both storing and querying — critical for
    cosine similarity scores to be meaningful across operations.

    Args:
        text: Any string — an atomic fact or a natural language query.

    Returns:
        384-element float list ready for Neo4j vector index storage/search.
    """
    return _get_model().encode(text, convert_to_tensor=False).tolist()
