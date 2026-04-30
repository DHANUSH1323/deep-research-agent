import asyncio
from functools import lru_cache

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import MCP_SERVER_URL

_MCP_SERVERS = {
    "deep-research-corpus": {
        "url": MCP_SERVER_URL,
        "transport": "streamable_http",
    }
}


@lru_cache(maxsize=1)
def _get_mcp_tools_by_name() -> dict[str, BaseTool]:
    """Initialize MCP clients and return a dict of tool name to MCP tool."""
    client = MultiServerMCPClient(_MCP_SERVERS)
    tools = asyncio.run(client.get_tools())
    return {tool.name: tool for tool in tools}

def get_mcp_tool(tool_name:str):
    """Get an MCP tool by name."""
    tools = _get_mcp_tools_by_name()
    if tool_name not in tools:
        raise ValueError(f"MCP tool '{tool_name}' not found. Available tools: {list(tools.keys())}")
    return tools[tool_name] 