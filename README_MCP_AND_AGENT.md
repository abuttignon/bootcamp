# Use Case 3: MCP & Agent

This use case demonstrates two things:

1. **MCP** — how to expose tools via the Model Context Protocol and let an LLM select and call them from natural language.
2. **Agent** — how to build a multi-turn conversational agent that routes between direct LLM answers and MCP tool calls.

**Goal:** Show how a standardized tool protocol works, and how even a relatively simple routing architecture can feel agent-like — while understanding its actual limitations.

---

## Table of Contents

- [Two Entry Points: What's the Difference?](#two-entry-points-whats-the-difference)
- [Available Tools](#available-tools)
- [Running the MCP Client](#running-the-mcp-client)
- [Running the Agent](#running-the-agent)
- [Agent Architecture](#agent-architecture)
- [What "Reasoning" Actually Happens](#what-reasoning-actually-happens)
- [Transport Modes](#transport-modes)
- [Key Files](#key-files)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Two Entry Points: What's the Difference?

| | `scripts/mcp_client.py` | `scripts/run_agent.py` |
|---|---|---|
| **Interaction** | Single-turn (one query → one answer) | Multi-turn chat with history |
| **Routing** | Always calls a tool | Router LLM decides: tool or direct answer |
| **Memory** | Stateless | Last 4 turns kept in context |
| **Transport** | `stdio` (default) or `sse` | SSE only (requires running server) |
| **LLM calls per turn** | 1 (tool selection) | 2–3 (router + optional sub-agent + responder) |

Use `mcp_client.py` to understand the MCP protocol itself.
Use `run_agent.py` to see how routing and tool use compose into an agent loop.

---

## Available Tools

Defined in `server/mcp_server.py` (FastMCP):

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_stock_price` | `symbol: str` | Current stock price via Yahoo Finance, fallback to `data/stocks_data.csv` |
| `compare_stocks` | `symbol1: str, symbol2: str` | Side-by-side price comparison with winner indication |
| `answer_recipe_query` | `query: str` | Full RAG pipeline: vector search → context → LLM answer |

`answer_recipe_query` runs the complete RAG pipeline from use case 2 as an MCP-callable tool.

---

## Running the MCP Client

### stdio mode (default — simplest, no server setup)

```bash
python scripts/mcp_client.py
```

The client spawns `server/mcp_server.py` as a subprocess automatically.

```
What is your query? → What is the stock price of Apple?
What is your query? → Compare Tesla and Microsoft
What is your query? → I have pasta and tomatoes, what can I cook?
```

### SSE mode (HTTP-based)

```bash
# .env:
MCP_TRANSPORT=sse
MCP_SERVER_URL=http://127.0.0.1:8000/sse

python server/mcp_server.py     # Terminal 1
python scripts/mcp_client.py    # Terminal 2
```

---

## Running the Agent

The agent requires SSE mode — the server must be running before starting the agent.

```bash
# .env:
MCP_TRANSPORT=sse
MCP_SERVER_URL=http://127.0.0.1:8000/sse

python server/mcp_server.py     # Terminal 1
python scripts/run_agent.py     # Terminal 2
```

Example session:
```
[agent] Ready. Type your message, or 'exit' to quit.
[you] What's the capital of France?
[router] decision: DIRECT_ANSWER
[assistant] The capital of France is Paris.
--------------------------------------------------
[you] I have chicken and rice, what can I make?
[router] decision: USE_TOOL
[router] tool_name: answer_recipe_query
[tool] raw_result: ...
[assistant] Here are some recipes you can make with chicken and rice: ...
```

---

## Agent Architecture

### Two-LLM Pipeline

The agent uses two (sometimes three) separate LLM calls per user turn — a router and a responder — plus optional MCP tool calls.

```
User input
    │
    ▼
[1] Router LLM
    (build_routing_prompt)
    Decides: DIRECT_ANSWER or USE_TOOL?
    │
    ├── DIRECT_ANSWER ──► [2a] Responder LLM
    │                          answers from knowledge
    │
    └── USE_TOOL ──────► [2b] Sub-agent LLM
                               (generate_mcp_tool_call_payload)
                               selects tool + infers args
                               │
                               ▼
                         MCP Server call
                               │
                               ▼
                         [2c] Responder LLM
                               formats tool output into answer

Final answer → stored in messages[]
```

### Step-by-Step Flow

**Startup**

1. Connects to the MCP server via SSE at `localhost:8000/sse`
2. Calls `retrieve_mcp_tools()` — lists all tools and builds a text description + set of tool names
3. Enters the chat loop

**Each user turn**

**Step 1 — Routing** (`run_agent.py:119–153`)
- Calls `build_routing_prompt()` with the tool list, user message, and last 4 turns of history
- Sends to the LLM → expects JSON with `decision`, `tool_name`, `reason_code`, `confidence`
- If parsing fails, retries once; if still broken, falls back to `DIRECT_ANSWER`

Routing decision schema:
```json
{
  "decision": "DIRECT_ANSWER | USE_TOOL",
  "tool_name": "string | null",
  "reason_code": "string",
  "reason": "string",
  "confidence": 0.0
}
```

**Step 2 — Recipe guardrail** (`run_agent.py:155–166`)
- Even if the router said `DIRECT_ANSWER`, a keyword check (`recipe`, `cook`, `chicken`, etc.) overrides it to force `USE_TOOL` with `answer_recipe_query`
- This prevents the LLM from hallucinating recipes from memory instead of using the RAG tool

**Step 3a — Direct answer** (`run_agent.py:170–177`)
- Simple prompt with history + user request → LLM replies directly

**Step 3b — Tool path** (`run_agent.py:178–203`)
- `run_mcp_request()` makes a third LLM call (`generate_mcp_tool_call_payload`) to infer the tool name and arguments as JSON
- Calls the MCP tool via `session.call_tool()`
- A fourth LLM call formats the raw tool output into a human-readable response

MCP call detail:
```
mcp_sub_agent.py              MCP server (mcp_server.py)
      │                               │
      │  call_tool(name, args)        │
      │ ────────────────────────────► │
      │                         executes tool function
      │  response (text)              │
      │ ◄──────────────────────────── │
```

---

## What "Reasoning" Actually Happens

This agent has no real reasoning or reflection loop. It's a single-pass pipeline, not an iterative reasoner. Here's what actually plays the "thinking" role:

| What looks like reasoning | Where it happens | Reality |
|---|---|---|
| "Should I use a tool?" | Router LLM call (`run_agent.py:125`) | One-shot LLM judgment, no iteration |
| "Which tool + what args?" | Sub-agent LLM call (`mcp_sub_agent.py:158`) | One-shot LLM inference |
| "What does the tool result mean?" | Responder LLM call (`run_agent.py:200`) | One-shot LLM formatting |
| Recipe guardrail | `should_force_recipe_tool()` (`run_agent.py:59`) | Hard-coded keyword check, not reasoning |

Each turn is: **route → (optionally call tool) → respond.** Then done. The agent never looks at its own output and thinks "was that good? should I try again?"

True agentic reasoning would involve:
- A **ReAct loop** — Reason, Act, Observe, repeat until done
- **Self-critique** — the agent evaluating its own response
- **Multi-step planning** — decomposing a task into sub-tasks

This agent is closer to a **smart router + tool dispatcher** than a reasoning agent. The LLM intelligence lives in each individual prompt, but the architecture itself has no reflection mechanism.

---

## Transport Modes

| Mode | Config | Use Case |
|------|--------|---------|
| `stdio` | Default (no `.env` change needed) | Local demo, single machine — client spawns the server |
| `sse` | `MCP_TRANSPORT=sse` + `MCP_SERVER_URL` | Distributed setup, or when using `run_agent.py` |

`run_agent.py` requires SSE — it connects to a running server and keeps the connection alive for the full chat session.

---

## Key Files

| File | Purpose |
|------|---------|
| `server/mcp_server.py` | FastMCP server with 3 tools |
| `scripts/mcp_client.py` | Single-turn MCP client — tool dispatcher |
| `scripts/run_agent.py` | Multi-turn agent with router, history, and guardrail |
| `src/agent_experiment/mcp_sub_agent.py` | Tool listing and MCP call execution |
| `src/agent_experiment/prompts.py` | Routing prompt template |

---

## Environment Variables

| Variable | Required | Value |
|----------|----------|-------|
| `OPENAI_API_KEY` | Yes | Used for all LLM calls in this use case |
| `ANTHROPIC_API_KEY` | Optional | If using Claude in the RAG tool |
| `MODEL_NAME` | Yes | LLM for routing and tool selection |
| `EMBEDDING_MODEL` | Yes | `text-embedding-3-small` |
| `MONGO_DB_URL` | Yes | Required for `answer_recipe_query` tool |
| `MONGO_COLLECTION_NAME_RAG` | Yes | `curated_chunks` |
| `MCP_TRANSPORT` | Optional | `stdio` (default) or `sse` |
| `MCP_SERVER_URL` | SSE only | `http://127.0.0.1:8000/sse` |

---

## Troubleshooting

**`Connection refused` when starting `run_agent.py`:**
- The MCP server must be running in SSE mode first: `python server/mcp_server.py`

**`answer_recipe_query` returns empty results:**
- RAG data must be set up first — see [README_RAG.md](README_RAG.md)

**Tool not found / wrong tool selected:**
- Check `OPENAI_API_KEY` in `.env`
- Rephrase the query to be more specific
