import asyncio
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()


def fetch_tool_identifier_prompt():
    tool_identifier_prompt = """

        You have been given access to the below MCP Server Tools

        {tools_description}

        You must identify the appropriate tool only from the above tools required to resolve the user query along with the arguments,

        {user_query}

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


async def generate_mcp_tool_call_payload(user_query: str, tools_description: str):
    """Ask the model to select one MCP tool and arguments."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL_NAME")
    client = openai.Client(api_key=api_key)

    tool_identifier_prompt = fetch_tool_identifier_prompt()
    tool_identifier_prompt = tool_identifier_prompt.format(
        user_query=user_query, tools_description=tools_description
    )

    response = client.responses.create(model=model, input=tool_identifier_prompt)

    raw = response.output_text
    raw = raw.replace("```json", "").replace("```", "")
    data = json.loads(raw)

    if isinstance(data["arguments"], str):
        args_list = [arg.strip() for arg in data["arguments"].split(",")]
        data["arguments"] = (
            {args_list[0]: args_list[1]} if len(args_list) > 1 else {args_list[0]: True}
        )

    return data


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


async def run_with_stdio(user_input: str):
    server_params = StdioServerParameters(
        command="python",
        args=["server/mcp_server.py"],
        cwd=os.getcwd(),
    )

    async with stdio_client(server_params) as (read, write):
        print("Connection established, creating session...")
        async with ClientSession(read, write) as session:
            await run_session(session, user_input)


async def run_with_sse(user_input: str):
    server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

    async with sse_client(server_url) as (read, write):
        print(f"Connected to running MCP server at {server_url}")
        async with ClientSession(read, write) as session:
            await run_session(session, user_input)


async def run_session(session: ClientSession, user_input: str):
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

    request_json = await generate_mcp_tool_call_payload(
        user_query=user_input, tools_description=tools_description
    )
    tool_name, arguments = validate_tool_request(request_json, available_tool_names)

    print(
        f"To execute the User Query: {user_input} - The identified tool is {tool_name}, "
        f"and the parameters required are {arguments}"
    )
    response_mcp = await session.call_tool(tool_name, arguments=arguments)
    print(f"{response_mcp.content[0].text}")  # type: ignore
    print("-" * 50)
    print("\n\n")


async def main(user_input: str):
    """Main function to handle MCP client session and tool execution.

    This function establishes a connection to the MCP server, initializes a session,
    lists available tools, identifies the appropriate tool using AI, and executes
    the identified tool with the provided arguments.
    """
    print("-" * 50)
    print("The User Input is : ", user_input)
    mode = os.getenv("MCP_TRANSPORT", "stdio").lower()

    try:
        if mode == "sse":
            await run_with_sse(user_input)
        else:
            await run_with_stdio(user_input)
    except Exception as e:
        print(f"[agent] Connection error: {str(e)}")
        print(f"[agent] Exception type: {type(e).__name__}")

        nested_exceptions = getattr(e, "exceptions", None)
        if nested_exceptions:
            print(f"[agent] Nested exceptions: {len(nested_exceptions)}")
            for idx, sub_exc in enumerate(nested_exceptions, start=1):
                print(
                    f"[agent] Sub-exception {idx}: {type(sub_exc).__name__}: {sub_exc}"
                )
                traceback.print_exception(type(sub_exc), sub_exc, sub_exc.__traceback__)

        print("[agent] Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    load_dotenv()
    while True:
        query = input("What is your query? → ")
        asyncio.run(main(query))
