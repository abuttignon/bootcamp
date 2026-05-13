# Use Case 2: RAG (Retrieval-Augmented Generation)

This use case demonstrates how grounding LLM responses in retrieved documents improves accuracy for recipe recommendations.

**Goal:** Show the difference between a "blind" LLM answer (prompt engineering) and a context-grounded answer (RAG) when asked about specific recipes.

---

## How It Works

```
user query
  → OpenAI embed (text-embedding-3-small, 1536 dims)
  → MongoDB dotProduct vector search (top-3 chunks from curated_chunks)
  → retrieved context injected into LLM prompt
  → LLM (Claude or OpenAI) → response
```

The `RAGService` handles the full pipeline: embedding the query, running the vector search, and calling the LLM with context.

---

## Prerequisites

1. Set up `recipes` collection in **MongoDB** 
2. **OpenAI API key** is required

### How it worksOne-time Setup

[recipes_data_preprocessing.ipynb](notebooks/recipes_data_preprocessing.ipynb) populates the MongoDB
In [recipes_rag.ipynb](notebooks/recipes_rag.ipynb) we can interact with the LLM.

```python
from src.recipes.rag.rag_service import RAGService
import os
from dotenv import load_dotenv

load_dotenv()

rag = RAGService(
    embedding_model=os.getenv("EMBEDDING_MODEL"),
    query_model=os.getenv("MODEL_NAME"),
    db_url=os.getenv("MONGO_DB_URL"),
    collection=os.getenv("MONGO_COLLECTION_NAME_RAG")
)

rag.setup()  # embeds documents + creates vector index (run once)
```
---

## Running the Demo

```bash
# Outside an IDE, use:
uv run --env-file .env jupyter lab
# Open notebooks/recipes_rag.ipynb
```

The notebook demonstrates:
- Loading and embedding recipe data
- Running vector queries
- Comparing RAG responses to prompt-only responses

### Script

```python
from src.recipes.rag.rag_service import RAGService

rag = RAGService(...)  # see setup above

response = rag.generate_response("I have chicken and lemon, what can I cook?")
print(response)
```

---

## MongoDB Vector Index

| Setting | Value        |
|---------|--------------|
| Collection | `recipes`    |
| Field | `embedding`  |
| Dimensions | `1536`       |
| Similarity | `dotProduct` |
| Results returned | Top 3        |

The index is created automatically during `RAGService.setup()`. You can verify it in MongoDB Atlas → Collections → `recipes` → Search Indexes.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/recipes/rag/rag_service.py` | Core RAG class |
| `notebooks/recipes_rag.ipynb` | Interactive demo notebook |
| `notebooks/recipes_data_preprocessing.ipynb` | Data loading walkthrough |

### `RAGService` API

```python
class RAGService:
    def setup()                        # One-time: embed docs + create index
    def generate_response(anfrage: str) -> str  # Main entry: query → response
    def query_database(anfrage: str) -> list    # Vector search only
    def embed_documents(documents)     # Bulk embed and write to MongoDB
```

---

## Environment Variables

| Variable | Required | Value                                |
|----------|----------|--------------------------------------|
| `OPENAI_API_KEY` | Yes | Always required for embeddings       |
| `MODEL_NAME` | Yes | e.g. `gpt-4o` |
| `EMBEDDING_MODEL` | Yes | `text-embedding-3-small`             |
| `MONGO_DB_URL` | Yes | MongoDB Atlas connection string      |
| `MONGO_DB_NAME` | Yes | `bootcamp`                           |
| `MONGO_COLLECTION_NAME_RAG` | Yes | `recipes`                            |

---
