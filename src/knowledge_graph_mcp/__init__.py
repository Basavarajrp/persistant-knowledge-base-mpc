import asyncio
from knowledge_graph_mcp.server import main as async_main

def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(async_main())

__all__ = ["main"]