import os
import asyncio
import json

import openai
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

from agent_experiment.mcp_sub_agent import retrieve_mcp_tools, run_mcp_request
from agent_experiment.prompts import build_routing_prompt


def format_history(messages: list[dict], max_turns: int = 4) -> str:
    if not messages:
        return ""

    recent = messages[-(max_turns * 2) :]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


def parse_router_response(raw_text: str, available_tool_names: set[str]) -> dict:
    cleaned = raw_text.strip().replace("```json", "").replace("```", "")
    data = json.loads(cleaned)
    if data.get("decision") not in {"DIRECT_ANSWER", "USE_TOOL"}:
        raise ValueError("Router returned invalid decision")

    if data["decision"] == "USE_TOOL":
        tool_name = data.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in available_tool_names:
            raise ValueError("Router selected invalid or unavailable tool_name")

    return data


def print_router_decision(routing_json: dict) -> None:
    decision = routing_json.get("decision", "UNKNOWN")
    tool_name = routing_json.get("tool_name")
    reason_code = routing_json.get("reason_code", "n/a")
    reason = routing_json.get("reason", "n/a")
    confidence = routing_json.get("confidence", "n/a")

    print("[router] decision:", decision)
    print("[router] tool_name:", tool_name)
    print("[router] reason_code:", reason_code)
    print("[router] reason:", reason)
    print("[router] confidence:", confidence)
    print("-" * 50)


def print_tool_result(tool_result: dict) -> None:
    print("[tool] selected:", tool_result.get("tool_name"))
    print("[tool] arguments:", tool_result.get("arguments"))
    print("[tool] raw_result:")
    print(tool_result.get("raw_result", ""))
    print("-" * 50)


def should_force_recipe_tool(user_request: str, available_tool_names: set[str]) -> bool:
    if "answer_recipe_query" not in available_tool_names:
        return False

    text = user_request.lower()
    recipe_keywords = {
        "recipe",
        "cook",
        "cooking",
        "ingredient",
        "ingredients",
        "meal",
        "dinner",
        "lunch",
        "breakfast",
        "vegetable",
        "vegetables",
        "chicken",
        "substitute",
        "nutrition",
    }
    return any(keyword in text for keyword in recipe_keywords)


async def main():
    # should hold a general CLI chat interface with temp memory of ongoing conversation
    # should be able to decide if it needs mcp tools.
    # If it does, should be able to establish connection to a running mcp server (localhost:8000/sse)

    # OpenAI initialization
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL_NAME")
    if not api_key or not model:
        raise ValueError("Missing OPENAI_API_KEY or MODEL_NAME in environment")

    client = openai.Client(api_key=api_key)

    messages = []

    # MCP initialization
    server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

    async with sse_client(server_url) as (read, write):
        print(f"Connected to running MCP server at {server_url}")
        async with ClientSession(read, write) as session:
            mcp_tools_description, available_tool_names = await retrieve_mcp_tools(
                session
            )
            print("[agent] Ready. Type your message, or 'exit' to quit.")
            print("-" * 50)

            while True:
                user_request = await asyncio.to_thread(input, "[you] ")
                if not user_request.strip():
                    continue
                if user_request.strip().lower() in {"exit", "quit"}:
                    print("[agent] Goodbye")
                    break

                history = format_history(messages)
                router_input = build_routing_prompt(
                    tools_description=mcp_tools_description,
                    user_request=user_request,
                    history=history,
                )

                route_response = client.responses.create(
                    model=model, input=router_input
                )
                try:
                    routing_json = parse_router_response(
                        route_response.output_text, available_tool_names
                    )
                except (json.JSONDecodeError, ValueError):
                    repair_prompt = (
                        "Your previous output did not match the required JSON contract. "
                        "Return one valid JSON object only, no markdown fences, no extra text.\n\n"
                        f"Original router prompt:\n{router_input}"
                    )
                    repair_response = client.responses.create(
                        model=model,
                        input=repair_prompt,
                    )
                    try:
                        routing_json = parse_router_response(
                            repair_response.output_text, available_tool_names
                        )
                    except (json.JSONDecodeError, ValueError):
                        routing_json = {
                            "decision": "DIRECT_ANSWER",
                            "reason_code": "ROUTER_PARSE_FAILURE",
                            "reason": "Could not parse route after one retry; falling back to direct answer",
                            "confidence": 0.0,
                            "tool_name": None,
                        }

                if routing_json[
                    "decision"
                ] == "DIRECT_ANSWER" and should_force_recipe_tool(
                    user_request, available_tool_names
                ):
                    routing_json = {
                        "decision": "USE_TOOL",
                        "tool_name": "answer_recipe_query",
                        "reason_code": "RECIPE_GUARDRAIL",
                        "reason": "Recipe request detected; forcing grounded recipe tool.",
                        "confidence": 1.0,
                    }

                print_router_decision(routing_json)

                if routing_json["decision"] == "DIRECT_ANSWER":
                    print("[agent] answering directly")
                    direct_prompt = (
                        f"Conversation so far:\n{history}\n\n"
                        f"Current user request:\n{user_request}"
                    )
                    response = client.responses.create(model=model, input=direct_prompt)
                    assistant_text = response.output_text
                elif routing_json["decision"] == "USE_TOOL":
                    print(f"[agent] using tool: {routing_json['tool_name']}")
                    tool_result = await run_mcp_request(
                        session=session,
                        query=user_request,
                        tools_description=mcp_tools_description,
                        available_tool_names=available_tool_names,
                        preferred_tool_name=routing_json["tool_name"],
                    )
                    print_tool_result(tool_result)
                    print("[agent] interpreting tool result for final response")

                    responder_prompt = (
                        f"Conversation so far:\n{history}\n\n"
                        f"Tool used:\n{tool_result['tool_name']}\n\n"
                        f"Tool arguments:\n{tool_result['arguments']}\n\n"
                        f"Tool raw output:\n{tool_result['raw_result']}\n\n"
                        f"Current user request:\n{user_request}\n\n"
                        "Write a concise, grounded response based only on the tool raw output. "
                        "If the tool output is insufficient, say so explicitly."
                    )

                    response = client.responses.create(
                        model=model, input=responder_prompt
                    )
                    assistant_text = response.output_text
                else:
                    assistant_text = "I could not determine how to proceed."

                messages.append({"role": "user", "content": user_request})
                messages.append({"role": "assistant", "content": assistant_text})
                print(f"[assistant] {assistant_text}")
                print("-" * 50)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
