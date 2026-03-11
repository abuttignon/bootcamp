def build_routing_prompt(
    tools_description: str, user_request: str, history: str = ""
) -> str:
    return f"""You are a routing component.

Your job is to decide whether the assistant should:
1. answer directly from existing knowledge/context, or
2. use one MCP tool.

Output JSON only.

Decision rules:
- Choose DIRECT_ANSWER only if the request can be answered without external data, tools, or side effects.
- Choose USE_TOOL if external data, execution, file access, current state, or system-grounded lookup is needed.
- Prefer DIRECT_ANSWER for general explanations, definitions, rewriting, summarization of provided text, and reasoning tasks that do not depend on current external state.
- Prefer USE_TOOL for database access, file retrieval, API queries, current status checks, actions, or user-specific system data.
- If the request is about recipes, cooking, ingredients, substitutions, meal ideas, or nutrition for dishes, choose USE_TOOL.
- If answer_recipe_query is available, use tool_name "answer_recipe_query" for recipe/cooking requests.
- If decision is USE_TOOL, tool_name must be exactly one available tool name.
- If decision is DIRECT_ANSWER, tool_name must be null.

Available tools:
{tools_description}

Recent conversation history (may be empty):
{history}

Current user request:
{user_request}

Return exactly:
{{
  "decision": "DIRECT_ANSWER" | "USE_TOOL",
  "tool_name": string | null,
  "reason_code": string,
  "reason": string,
  "confidence": number
}}"""
