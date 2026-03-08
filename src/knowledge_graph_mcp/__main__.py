"""
__main__.py — Allows running as: python -m knowledge_graph_mcp
Useful for debugging outside of uv.
"""

from knowledge_graph_mcp.server import main
import asyncio

asyncio.run(main())
