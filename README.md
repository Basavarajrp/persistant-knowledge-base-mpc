# Knowledge Graph MCP - Quick Start

## ⚡ Quick Setup

**All commands run from PROJECT ROOT** (`knowledge-graph-mcp-v2/`), NOT from `src/` folder:

```bash
# 1. Configure (PROJECT ROOT)
cd /path/to/knowledge-graph-mcp-v2
cp .env-complete .env
# Edit .env: NEO4J_PASSWORD=your_secure_password_here

# 2. Start database (PROJECT ROOT)
docker compose up -d

# 3. Install Python deps (PROJECT ROOT)
uv sync

# 4. Run MCP server (PROJECT ROOT)
uv run knowledge-graph-mcp
```

Done! Neo4j in Docker, MCP server running locally, both on same network.

---

## ⚠️ IMPORTANT: Folder Structure

```
knowledge-graph-mcp-v2/          ← PROJECT ROOT (run commands here)
├── .env                         ← Config file
├── .env-complete
├── docker-compose.yml
├── pyproject.toml
├── README-QUICK.md
├── src/                         ← Source code (don't run commands here)
│   └── knowledge_graph_mcp/
│       ├── __init__.py
│       ├── server.py
│       └── ...
```

**❌ Wrong:**
```bash
cd knowledge-graph-mcp-v2/src
uv run knowledge-graph-mcp  # ❌ Won't work
```

**✅ Correct:**
```bash
cd knowledge-graph-mcp-v2
uv run knowledge-graph-mcp  # ✅ Works
```

---

## 🗄️ Database Access

| Service | URL | Credentials |
|---------|-----|-------------|
| **Neo4j Browser** | http://localhost:7474 | `neo4j` / `${NEO4J_PASSWORD}` |
| **Bolt (Code)** | `bolt://localhost:7687` | Same as above |

**Import to .env:**
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password_here
```

---

## 🔗 Connect to Claude Desktop

1. **Edit Claude config** (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/knowledge-graph-mcp-v2",
        "run",
        "knowledge-graph-mcp"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_secure_password_here"
      }
    }
  }
}
```

2. **Restart Claude Desktop** → it connects to the local MCP server process

3. **Use in Claude:**
```
"Remember: API rate limit is 1000 req/min"
"What did I tell you about the API?"
```

---

## 🔗 Connect to Cursor

**Option 1: Via MCP Server (Recommended)**
- Same as Claude Desktop config
- Edit Cursor MCP settings (see Cursor docs)

**Option 2: Via Claude API**
- Use Claude extension in Cursor
- MCP server runs locally, Cursor connects via stdio

---

## 📊 Visualize Data

**Neo4j Browser:**
```bash
# Open: http://localhost:7474
# Login: neo4j / your_password
# Run Cypher:

MATCH (p:Profile) RETURN p LIMIT 10
MATCH (p:Profile)-[:HAS_CATEGORY]->(c:Category)-[:HAS_FACT]->(f:Fact) RETURN *
```

---

## 🛠️ MCP Server Tools

| Tool | Input | Purpose |
|------|-------|---------|
| `store_fact` | `fact`, `profile_id`, `category` | Store knowledge with embeddings |
| `query_knowledge` | `query`, `profile_id`, `top_k` | Search semantically |
| `list_profiles` | - | Find all profiles |
| `list_categories` | `profile_id` | Find categories in profile |

---

## 📁 Project Files

```
knowledge-graph-mcp-v2/
├── .env                      ← Config (edit this)
├── docker-compose.yml        ← DB setup
├── pyproject.toml           ← Dependencies
├── src/                     ← MCP server code
├── README.md                ← Full docs
├── ARCHITECTURE_GUIDE.md    ← Deep dive
└── SETUP_GUIDE.md           ← Detailed setup
```

---

## ✅ Verify Setup

```bash
# 1. Neo4j running?
docker ps | grep neo4j
# Should show status: "healthy"

# 2. MCP server running?
# Terminal should show: "Server running on stdio"

# 3. Test in Claude Desktop:
# Try: "Remember: test data 123"
# Then: "What did I tell you?"
# Should work immediately
```

---

## 🔑 Important Ports & Credentials

- **Neo4j Port:** `7687` (Bolt protocol for code)
- **Neo4j Browser:** `7474` (UI visualization)
- **Username:** `neo4j` (fixed)
- **Password:** `${NEO4J_PASSWORD}` (set in .env)
- **Embedding Model:** `all-MiniLM-L6-v2` (auto-downloads, cached)

---

## 🚨 Common Issues

| Problem | Solution |
|---------|----------|
| "Connection refused" | Wait 30s, check: `docker logs knowledge-graph-neo4j` |
| "Auth failed" | Verify `NEO4J_PASSWORD` in `.env` |
| "Port 7687 in use" | Change in `docker-compose.yml`: `"7688:7687"` |
| "Slow first embedding" | Normal (~2 min), model caches after |

---

## 📖 Need More?

- **Full docs:** See this [blog](https://basavarajrp.hashnode.dev/llm-persistant-memory)
