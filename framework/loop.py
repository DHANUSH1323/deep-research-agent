from anthropic.types import Message
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_anthropic_client
from framework.tools import get_all_schemas, get_tool

def run_agent_loop(
        messages: list[dict],
        system: str = "",
        max_iterations: int = 10,
        model: str = SUBAGENT_MODEL,
) -> Message:
    """Run the agent loop until a final answer is produced or max iterations is reached."""
    client = get_anthropic_client()
    tool_list = get_all_schemas()
    for iteration in range(max_iterations):
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system
        if tool_list:
            kwargs["tools"] = tool_list
        response = client.messages.create(**kwargs)

        messages.append({"role": "assistant",
                         "content": response.content})
        
        if response.stop_reason == "end_turn":
            return response
        elif response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_func = get_tool(block.name)
                    result = tool_func(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
            messages.append({"role": "user", "content": tool_results})
            continue
        else:
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")
    raise RuntimeError(f"Agent loop exceeded max_iterations={max_iterations}")