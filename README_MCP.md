# Use Case 3: MCP & Agent

This use case demonstrates how LLMs can invoke external tools dynamically using the Model Context Protocol (MCP), and how an agent layer can route between direct answers and tool calls.

**Goal:** Show how to expose tools via a standardized protocol and let an LLM decide which tool to use and partially control the routing logic programmatically.

---

## Table of Contents

- [MCP: How It Works](#mcp-how-it-works)
- [Available Tools](#available-tools)
- [Running the MCP Demo](#running-the-mcp-demo)
- [Agent Orchestration](#agent-orchestration)
- [Transport Modes](#transport-modes)
- [Key Files](#key-files)
- [Environment Variables](#environment-variables)

---

## MCP: How It Works

```
user query (natural language)
  → MCP client sends query + tool descriptions to OpenAI
  → OpenAI selects tool + arguments (JSON)
  → MCP client calls selected tool on FastMCP server
  → tool result returned to user or to LLM for further processing
```

The server and client are separate processes. The client uses OpenAI **only for tool selection** — the actual execution happens via the MCP protocol on the server side.

---

## Available Tools

Defined in `server/mcp_server.py` (FastMCP):

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_stock_price` | `symbol: str` | Current stock price via Yahoo Finance, fallback to `data/stocks_data.csv` |
| `compare_stocks` | `symbol1: str, symbol2: str` | Side-by-side price comparison with winner indication |
| `answer_recipe_query` | `query: str` | Full RAG pipeline: vector search → context → LLM answer |

The `answer_recipe_query` tool runs the complete RAG pipeline from use case 2, making it available as an MCP-callable tool.

---

## Running the MCP Demo

### stdio mode (default, simplest)

```bash
# Terminal: run the interactive client
python scripts/mcp_client.py

# The client starts the MCP server server/mcp_server.py 
```

You will be prompted:
```
What is your query? → What is the stock price of Apple?
What is your query? → Compare Tesla and Microsoft
What is your query? → I have pasta and tomatoes, what can I cook?
```

The client automatically identifies the appropriate tool and executes it.

### SSE mode (HTTP-based, for distributed setups)

```bash
# Set in .env:
MCP_TRANSPORT=sse
MCP_SERVER_URL=http://127.0.0.1:8000/sse

python server/mcp_server.py
python scripts/mcp_client.py
```

---

## Agent Orchestration

`src/agent_experiment/` contains a higher-level agent that routes between direct answers and tool use.

**Flow:**
```
user request
  → routing prompt → OpenAI (decide: DIRECT_ANSWER or USE_TOOL)
  → if DIRECT_ANSWER: respond from model knowledge
  → if USE_TOOL: identify tool → call via MCP → return result
```

**Routing decision schema:**
```json
{
  "decision": "DIRECT_ANSWER | USE_TOOL",
  "tool_name": "string | null",
  "reason_code": "string",
  "reason": "string",
  "confidence": 0.0
}
```

**Decision rules:**
- `DIRECT_ANSWER`: factual/general knowledge, no side effects needed
- `USE_TOOL`: requires external data, real-time lookup, or system execution

### Running the Agent

```bash
# Set in .env:
MCP_TRANSPORT=sse
MCP_SERVER_URL=http://127.0.0.1:8000/sse

# Requires MCP server running in SSE mode first
python server/mcp_server.py 

# Run agent
python -m src.agent_experiment.main
```

---

## Transport Modes

| Mode | Config                                 | Use Case |
|------|----------------------------------------|---------|
| `stdio` | Default                                | Local demo, single machine |
| `sse` | `MCP_TRANSPORT=sse` + `MCP_SERVER_URL` | Distributed, HTTP-accessible server |

---

## Key Files

| File | Purpose |
|------|---------|
| `server/mcp_server.py` | FastMCP server with 3 tools |
| `scripts/mcp_client.py` | Interactive client with OpenAI tool selection |
| `src/agent_experiment/main.py` | Agent routing loop |
| `src/agent_experiment/mcp_sub_agent.py` | MCP tool identification and execution |
| `src/agent_experiment/prompts.py` | Routing prompt template |

---

## Environment Variables

| Variable | Required | Value |
|----------|----------|-------|
| `OPENAI_API_KEY` | Yes | Used for tool selection |
| `ANTHROPIC_API_KEY` | Optional | If using Claude in RAG tool |
| `MODEL_NAME` | Yes | LLM for the `answer_recipe_query` tool |
| `EMBEDDING_MODEL` | Yes | `text-embedding-3-small` |
| `MONGO_DB_URL` | Yes | Required for `answer_recipe_query` tool |
| `MONGO_COLLECTION_NAME_RAG` | Yes | `curated_chunks` |
| `MCP_TRANSPORT` | Optional | `stdio` (default) or `sse` |
| `MCP_SERVER_URL` | SSE only | `http://127.0.0.1:8000/sse` |

---

## Troubleshooting

**`Connection refused` when starting client:**
- Make sure the MCP server is running in Terminal 1 before starting the client

**`answer_recipe_query` returns empty results:**
- RAG data must be set up first — see [README_RAG.md](README_RAG.md) for setup instructions

**Tool not found / wrong tool selected:**
- The tool selection uses OpenAI — check `OPENAI_API_KEY` in `.env`
- Rephrase the query to be more specific about what you want
