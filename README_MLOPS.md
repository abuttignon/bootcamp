# Use Case 4: MLOps Pipeline

This use case demonstrates a production-style MLOps workflow using MLflow to orchestrate a multi-stage document ingestion and prompt evaluation pipeline.

**Goal:** Show how to track, cache, and reproduce data preparation and prompt engineering experiments with MLflow.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Running the Pipeline](#running-the-pipeline)
- [Stage Details](#stage-details)
- [Prompt Engineering Pipeline](#prompt-engineering-pipeline)
- [MongoDB Collections](#mongodb-collections)
- [MLflow Tracking](#mlflow-tracking)
- [MLproject Reference](#mlproject-reference)

---

## Pipeline Overview

The pipeline transforms raw source documents into MongoDB-ready vector chunks, then registers and evaluates prompt versions — all tracked and cached by MLflow.

```
data/external/      Stage 1       data/interim/     Stage 2     data/processed/     Stage 3      MongoDB
  ├── *.pdf    ─────────────►  NormalizedDocument  ─────────►  ProcessedChunk   ─────────────►  curated_chunks
  └── **/*.txt   normalize         (JSON)              chunk        (JSONL)             upload

                Stage 4                      Stage 5 + 6
              MongoDB  ◄────────────    MLflow Prompt Registry
            (embeddings)  retrieval    (register + evaluate)
```

**Six sequential stages**, orchestrated by `run_mlflow_pipeline.py` with git-commit-based caching.

---

## Running the Pipeline

```bash
mlflow server # not necessary for pipeline but starts GUI

# Full pipeline with caching (only re-runs changed stages)
python scripts/run_mlflow_pipeline.py

# Force re-run all stages (ignore cache)
python scripts/run_mlflow_pipeline.py --use-cache False
```

**Caching logic:** Before running a stage, `_already_ran()` checks MLflow for a completed run with the same entry point name and git commit hash. If found, that run is reused — stages only re-execute when source code changes (new git commit).

**When to disable cache:**
- After editing pipeline stage code without committing
- To force a full fresh run for debugging or data updates

---

## Stage Details

### Stage 1a: `normalize_pdf_to_json` — PDF → NormalizedDocument

- Reads PDF files from `data/external/` recursively
- Uses PyMuPDF (`fitz`) to extract text page-by-page
- Detects chapter hints via `"Chapter N"` pattern
- Produces one `NormalizedDocument` JSON per file in `data/interim/`

### Stage 1b: `normalize_txt_to_json` — TXT → NormalizedDocument

- Reads `.txt` files from `data/external/` recursively
- Treats each file as a single `TextSegment`
- Uses filename (without extension) as section path

**Output format** (`data/interim/*.json`):
```json
{
  "doc_id": "sha1-based-id",
  "title": "filename",
  "source_path": "data/external/...",
  "source_type": "pdf | txt",
  "folder_context": ["dir", "subdir"],
  "chapter_hints": ["Chapter 1"],
  "page_count": 42,
  "segments": [
    { "text": "...", "section_path": ["Chapter 1"], "page": 1 }
  ]
}
```

### Stage 2: `normalized_to_chunks` — Chunk Documents

**Chunker:** `StructureAwareChunker` in `src/ml_ops_experiment/ingestion/common/chunker.py`
- Chunk size: **800 characters**
- Overlap: **120 characters** (step = 680)
- Sliding window over each `TextSegment.text`
- Inherits `section_path`, `page`, and `chapter_hints` from parent segment

**Chunk ID:** SHA1 of `f"{doc_id}|{'/'.join(section_path)}|{start}:{end}|{text[:80]}"` — deterministic, idempotent

**Output format** (`data/processed/*.jsonl`):
```json
{"chunk_id": "abc123", "doc_id": "...", "chunk_index": 0, "text": "...", "source_type": "pdf", "section_path": ["Chapter 1"], "page_start": 1, "page_end": 1, "token_estimate": 42}
```

### Stage 3: `chunks_to_db` — Upload to MongoDB

**Command:**
```bash
python -m src.ml_ops_experiment.ingestion.upload_jsonl_to_db
```

1. Reads all JSONL files from `data/processed/`
2. Deserializes each line as `ProcessedChunk`
3. Bulk upserts using `UpdateOne({"chunk_id": ...}, {"$set": ...}, upsert=True)`
4. Creates indices after upload:
   - `doc_id`
   - `source_type`
   - `(doc_id, chunk_index)` compound

**MLflow metrics logged:** `files_seen`, `lines_read`, `valid_docs`, `inserted_count`, `updated_count`, `parse_errors`

> **Note:** Embeddings are NOT added in this stage. Stage 4 handles embedding generation.

### Stage 4: `indexing` — Generate Embeddings

- Reads chunks from MongoDB (`curated_chunks` collection)
- Generates embeddings using OpenAI `text-embedding-3-small` (1536 dims)
- Writes vectors back to MongoDB
- Creates the `dotProduct` vector search index

**MongoDB vector index:**
- Collection: `curated_chunks`
- Field: `embedding`
- Dimensions: `1536`
- Similarity: `dotProduct`

### Stage 5: `register_prompts` — Register to MLflow Prompt Registry

**Command:**

- Registers 6 prompt versions (v1–v6) to MLflow Prompt Registry
- Logs registration summary as MLflow artifact

```python
prompt = mlflow.genai.register_prompt(
    name="prompt_v6",
    template=SYSTEM_INSTRUCTIONS_6,
    commit_message="Enhanced prompt with advanced techniques",
    tags={...}
)
```

### Stage 6: `evaluate_prompts` — Evaluate Prompts

- Loads prompts from registry by prompt name (`name_or_uri`)
- Evaluates against structured dataset
- Logs evaluation metrics and results as MLflow artifacts

**Evaluation metrics:**
1. **Must-Include Score (0–1):** Percentage of required terms/concepts found in the answer
2. **Forbidden Score (0 or 1):** Whether the answer avoided incorrect/forbidden terms
3. **Combined Score (0–1):** Weighted average (70% must-include + 30% forbidden)

---

## Prompt Engineering Pipeline

The prompt engineering stages follow [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/) best practices:

### Versioned Prompts

Six system prompt versions in `data/raw/prompts_python.py`, demonstrating progressive prompt engineering:

| Version | Technique |
|---------|-----------|
| v1 | Basic instruction |
| v2 | More detailed, metric-aware |
| v3 | Persona + structured output |
| v4 | Self-validation in 3 steps |
| v5 | Two-variant approach (best use vs. rejection) |
| v6 | Strict JSON schema, context verification, and insufficiency reporting |

### Name-Based Lifecycle

```python
# Register immutable prompt versions
prompt = mlflow.genai.register_prompt(
    name="python_assistant_v6",
    template=SYSTEM_INSTRUCTIONS_6,
    commit_message="Add strict JSON schema with context verification and code examples",
    tags={"version_type": "v6"}
)

# Load prompt for evaluation by name
loaded_prompt = mlflow.genai.load_prompt(name_or_uri="python_assistant_v6")
```

### Loading a Registered Prompt

Prompt loading is implemented in `src/ml_ops_experiment/prompt_engineering/evaluate_prompts.py`.

---

## MongoDB Collections

| Collection | Purpose | Created By |
|------------|---------|------------|
| `curated_chunks` | ProcessedChunks with embeddings | Stage 3 (upload) + Stage 4 (indexing) |

**Vector Search Index on `curated_chunks`:**
- Field: `embedding`
- Dimensions: `1536` (OpenAI `text-embedding-3-small`)
- Similarity: `dotProduct`

---

## MLflow Tracking

All stages are tracked in MLflow (default: SQLite locally in `mlruns/`).

```bash
# View pipeline runs and metrics
mlflow server
# Open http://localhost:5000
```

**Run parameters tracked per stage:**
- `mlflow_pipeline_id` — parent run ID for grouping
- Stage-specific parameters (paths, model names, etc.)

---

## MLproject Reference

Defined in `MLproject` at the project root:

| Entry Point | Command |
|-------------|---------|
| `normalize_pdf_to_json` | `convert_pdf_to_json.py` |
| `normalize_txt_to_json` | `convert_txt_to_json.py` |
| `normalized_to_chunks` | `convert_json_to_jsonl.py` |
| `chunks_to_db` | `upload_jsonl_to_db.py` |
| `indexing` | `indexing.py` |
| `retrieval` | `retrieval.py` |
| `register_prompts` | `register_prompts.py` |
| `evaluate_prompts` | `evaluate_prompts.py` |
| `main` | `run_mlflow_pipeline.py` (with caching) |

---
