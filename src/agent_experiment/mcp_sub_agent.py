import json
import os

import openai
from mcp import ClientSession


async def run_mcp_request(
    session: ClientSession,
    query: str,
    tools_description: str,
    available_tool_names: set[str],
    preferred_tool_name: str | None = None,
) -> dict:
    return await run_session(
        session=session,
        query=query,
        tools_description=tools_description,
        available_tool_names=available_tool_names,
        preferred_tool_name=preferred_tool_name,
    )


def fetch_tool_identifier_prompt():
    tool_identifier_prompt = """

        You have been given access to the below MCP Server Tools

        {tools_description}

        You must identify the appropriate tool only from the above tools required to resolve the user query along with the arguments.

        If preferred_tool_name is provided, you must use exactly that tool and only infer arguments.

        {user_query}

        preferred_tool_name: {preferred_tool_name}

        Return JSON only (no markdown fences) in the format below.

        {{
            "user_query": "User Query",
            "tool_identified": "Tool Name",
            "arguments": {{"arg1": "value"}}
        }}

        Example:

        User Query: What is the weather in Bengaluru?

        Your Response:
        {{
            "user_query": "What is the weather in Bengaluru?",
            "tool_identified": "get_weather",
            "arguments": {{"location":"BLR"}}
        }}

        """
    return tool_identifier_prompt


def normalize_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        # If it’s a str, it splits on commas, trims whitespace
        args_list = [arg.strip() for arg in arguments.split(",") if arg.strip()]
        if len(args_list) > 1:
            # If there are 2+ tokens, returns {first: second}
            return {args_list[0]: args_list[1]}
        if len(args_list) == 1:
            # If there’s exactly 1 token, returns {token: True}
            return {args_list[0]: True}
        return {}

    return {}


def validate_tool_request(
    request_json: dict, available_tool_names: set[str]
) -> tuple[str, dict]:
    tool_name = request_json.get("tool_identified", "")  # extract tools
    arguments = normalize_arguments(request_json.get("arguments", {}))

    if tool_name not in available_tool_names:
        raise ValueError(
            f"Model selected unknown tool '{tool_name}'. Available tools: {sorted(available_tool_names)}"
        )

    return tool_name, arguments


async def retrieve_mcp_tools(session: ClientSession) -> tuple[str, set[str]]:
    print("[agent] Session created, initializing...")
    await session.initialize()
    print("[agent] MCP session initialized")

    tools = await session.list_tools()
    tools_description = ""
    available_tool_names = set()
    for each_tool in tools.tools:
        available_tool_names.add(each_tool.name)
        current_tool_description = "Tool - " + each_tool.name + ":" + "\n"
        current_tool_description += each_tool.description + "\n"  # type: ignore
        tools_description += current_tool_description + "\n"

    return tools_description, available_tool_names


async def run_session(
    session: ClientSession,
    query: str,
    tools_description: str,
    available_tool_names: set[str],
    preferred_tool_name: str | None = None,
) -> dict:
    request_json = await generate_mcp_tool_call_payload(
        user_query=query,
        tools_description=tools_description,
        preferred_tool_name=preferred_tool_name,
    )
    tool_name, arguments = validate_tool_request(request_json, available_tool_names)

    print(
        f"To execute the User Query: {query} - The identified tool is {tool_name}, "
        f"and the parameters required are {arguments}"
    )
    response = await session.call_tool(tool_name, arguments=arguments)
    tool_text = response.content[0].text  # type: ignore
    print(f"{tool_text}")
    print("-" * 50)
    print("\n\n")
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "raw_result": tool_text,
    }


async def generate_mcp_tool_call_payload(
    user_query: str,
    tools_description: str,
    preferred_tool_name: str | None = None,
):
    """Ask the model to select one MCP tool and arguments."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL_NAME")
    client = openai.Client(api_key=api_key)

    tool_identifier_prompt = fetch_tool_identifier_prompt()
    tool_identifier_prompt = tool_identifier_prompt.format(
        user_query=user_query,
        tools_description=tools_description,
        preferred_tool_name=preferred_tool_name,
    )

    response = client.responses.create(model=model, input=tool_identifier_prompt)

    raw = response.output_text
    raw = raw.replace("```json", "").replace("```", "")
    data = json.loads(raw)

    if isinstance(data.get("arguments"), str):
        args_list = [arg.strip() for arg in data["arguments"].split(",")]
        data["arguments"] = (
            {args_list[0]: args_list[1]} if len(args_list) > 1 else {args_list[0]: True}
        )

    if preferred_tool_name:
        data["tool_identified"] = preferred_tool_name

    return data
