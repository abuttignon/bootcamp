# Use Case 1: Prompt Engineering

This use case demonstrates how system prompt design alone shapes LLM behavior for recipe recommendations — no retrieval, no tools, just the model.

**Goal:** Show how iterative prompt engineering progressively improves response quality, structure, and reliability.

---

## How It Works

```
user query → PromptEngineeringService → LLM (Claude or OpenAI) → response
```

The service builds a LangChain chain:
```
SystemPrompt | HumanPrompt | LLM | StrOutputParser
```

Model selection is automatic based on `MODEL_NAME`:
- `claude-*` → `ChatAnthropic`
- `gpt-*` / `o1-*` → `ChatOpenAI`

---

## Prompt Versions

Six system prompt versions in `data/raw/prompts_recipes.py`, each demonstrating a different technique:

| Version | Technique | What changed |
|---------|-----------|--------------|
| v1 | Basic instruction | Simple "be a helpful chef" prompt |
| v2 | Detailed constraints | Added metric-aware instructions |
| v3 | Persona + structure | Chef persona with structured output |
| v4 | Self-validation | 3-step self-check before answering |
| v5 | Dual variant | Two outputs: best use of ingredients vs. refusal with suggestions |
| v6 | Strict JSON schema | Full JSON schema enforcement, ingredient tracking, impossibility detection |

v6 (latest) enforces a strict JSON output schema — the model validates ingredients, tracks what's used, and explicitly flags when a recipe is not possible with the given inputs.

---

## Running the Demo

### Notebook (recommended)

```bash
# Outside an IDE, use:
uv run --env-file .env jupyter lab
# Open notebooks/recipes_prompt_engineering.ipynb
```

The notebook walks through all 6 prompt versions interactively, letting you compare responses.

### Script

```python
from src.recipes.prompt_engineering.prompt_engineering_service import PromptEngineeringService
from data.raw.prompts_recipes import SYSTEM_INSTRUCTIONS_6

service = PromptEngineeringService(
    model="claude-sonnet-4-6",   # or gpt-4o
    system_prompt=SYSTEM_INSTRUCTIONS_6
)

response = service.answer_anfrage("I have eggs, flour, and butter. What can I cook?")
print(response)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/recipes/prompt_engineering/prompt_engineering_service.py` | LangChain service wrapper |
| `data/raw/prompts_recipes.py` | System prompts v1–v6 |
| `notebooks/recipes_prompt_engineering.ipynb` | Interactive demo notebook |

---

## Environment Variables

| Variable | Required | Value |
|----------|----------|-------|
| `OPENAI_API_KEY` | If using OpenAI | — |
| `ANTHROPIC_API_KEY` | If using Claude | — |
| `MODEL_NAME` | Yes | e.g. `claude-sonnet-4-6` or `gpt-4o` |
