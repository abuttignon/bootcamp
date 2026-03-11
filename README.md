# MLOps Bootcamp: GenAI Patterns

A hands-on bootcamp demonstrating four GenAI/MLOps patterns applied to recipe recommendations and document Q&A using Python.

## Use Cases

| # | Use Case | Technology | Notebook / Entry Point |
|---|----------|-----------|----------------------|
| 1 | Recipe recommendations via **Prompt Engineering** | LangChain + Claude/OpenAI | [`notebooks/recipes_prompt_engineering.ipynb`](notebooks/recipes_prompt_engineering.ipynb) |
| 2 | Recipe recommendations via **RAG** | MongoDB Vector Search + OpenAI Embeddings | [`notebooks/recipes_rag.ipynb`](notebooks/recipes_rag.ipynb) |
| 3 | Stock & Recipe Q&A via **MCP + Agent** | FastMCP + OpenAI tool selection | [`scripts/mcp_client.py`](scripts/mcp_client.py) |
| 4 | Document ingestion & prompt evaluation via **MLOps** | MLflow Pipelines + Prompt Registry | [`scripts/run_mlflow_pipeline.py`](scripts/run_mlflow_pipeline.py) |

Detailed documentation per use case:

- [Prompt Engineering README](README_PROMPT_ENGINEERING.md)
- [RAG README](README_RAG.md)
- [MCP & Agent README](README_MCP.md)
- [MLOps Pipeline README](README_MLOPS.md)

---

## Quick Start

### Prerequisites

- **Python 3.13**
- **[uv](https://github.com/astral-sh/uv)** package manager
- **MongoDB Atlas account** (free tier works) — see [MongoDB Setup](#mongodb-setup)
- **API keys:** OpenAI (required for embeddings), Anthropic (optional, for Claude models)

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd bootcamp_handson_genai_mlops
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials (see Environment Variables section)
```

---

## MongoDB Setup

MongoDB is used for storing recipe data and processed document chunks with vector embeddings.

### 1. Create a Free Cluster on MongoDB Atlas

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) and sign up
2. Create a new cluster — choose **Free tier (M0)**
3. Select cloud provider and region, name it (e.g. `bootcamp-cluster`), click **Create Deployment**

### 2. Create a Database User

1. **Security → Database Access → Add New Database User**
2. Authentication: **Password** — username `user`, password `notasecret`
3. Privileges: **Read and write to any database**

### 3. Configure Network Access

1. **Security → Network Access → Add IP Address**
2. For the bootcamp: **Allow Access from Anywhere** (`0.0.0.0/0`)

### 4. Get Your Connection String

1. **Database → Connect → Connect your application**
2. Driver: Python 3.12+. Copy the string:
   ```
   mongodb+srv://user:notasecret@bootcamp-cluster.xxxxx.mongodb.net/bootcamp?retryWrites=true&w=majority
   ```

### 5. Configure `.env`

```bash
MONGO_DB_URL=mongodb+srv://user:notasecret@bootcamp-cluster.xxxxx.mongodb.net/bootcamp?retryWrites=true&w=majority
MONGO_DB_NAME=bootcamp
MONGO_COLLECTION_NAME_RECIPES=recipes
MONGO_COLLECTION_NAME_RAG=curated_chunks
```

### 6. Verify Connection

```bash
python -c "
from pymongo import MongoClient; import os; from dotenv import load_dotenv
load_dotenv()
client = MongoClient(os.getenv('MONGO_DB_URL'))
print('Connected to MongoDB:', client.server_info()['version'])
"
```

---

## Environment Variables

Create `.env` from the template:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | GPT models + embeddings (RAG always requires this) |
| `ANTHROPIC_API_KEY` | Optional | Claude models |
| `MODEL_NAME` | Yes | Active LLM, e.g. `claude-sonnet-4-6` or `gpt-4o` |
| `EMBEDDING_MODEL` | Yes | `text-embedding-3-small` |
| `MONGO_DB_URL` | Yes | MongoDB Atlas connection string |
| `MONGO_DB_NAME` | Yes | Database name (default: `bootcamp`) |
| `MONGO_COLLECTION_NAME_RECIPES` | Yes | Recipe source collection (`recipes`) |
| `MONGO_COLLECTION_NAME_RAG` | Yes | Chunked RAG collection (`curated_chunks`) |
| `MCP_TRANSPORT` | MCP only | `stdio` or `sse` |
| `MCP_SERVER_URL` | MCP/SSE only | `http://127.0.0.1:8000/sse` |

**Model selection:** `claude-` prefix → ChatAnthropic, `gpt-`/`o1-` prefix → ChatOpenAI

---

## Project Structure

```
bootcamp_handson_genai_mlops/
├── data/
│   ├── external/          # Raw PDFs and TXT files (input)
│   ├── interim/           # NormalizedDocument JSON (Stage 1 output)
│   ├── processed/         # ProcessedChunk JSONL (Stage 2 output)
│   ├── retrieved/         # Vector search results (Stage 5 output)
│   ├── eval/              # Evaluation datasets
│   └── raw/
│       ├── prompts_recipes.py   # Versioned recipe system prompts (v1–v6)
│       └── prompts_python.py    # Python documentation prompts
├── src/
│   ├── recipes/
│   │   ├── prompt_engineering/
│   │   │   └── prompt_engineering_service.py
│   │   └── rag/
│   │       └── rag_service.py
│   ├── agent_experiment/        # Agent orchestration (WIP)
│   │   ├── main.py
│   │   ├── mcp_sub_agent.py
│   │   └── prompts.py
│   └── ml_ops_experiment/       # MLflow pipeline components
│       ├── ingestion/           # Normalize, chunk, upload stages
│       ├── indexing/            # Embedding generation
│       ├── retrieval/           # Vector search
│       └── prompt_engineering/  # Register & evaluate prompts
├── server/
│   └── mcp_server.py            # FastMCP server (stock + recipe tools)
├── scripts/
│   ├── run_mlflow_pipeline.py   # Pipeline orchestrator
│   └── mcp_client.py            # Interactive MCP client
├── notebooks/
│   ├── recipes_rag.ipynb
│   ├── recipes_prompt_engineering.ipynb
│   └── recipes_data_preprocessing.ipynb
├── MLproject                    # MLflow pipeline definition
└── .env.example                 # Environment template
```

---

## Common Commands

```bash
# Launch Jupyter for notebooks (use cases 1 & 2)
jupyter lab

# MCP demo (use case 3)
  python scripts/mcp_client.py       # Terminal 2: interactive client

# MLflow pipeline (use case 4)
mlflow server  # View MLflow UI http://localhost:5000
python scripts/run_mlflow_pipeline.py              # Run with caching
python scripts/run_mlflow_pipeline.py --use-cache False  # Force re-run
```

---