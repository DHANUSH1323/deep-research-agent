"""Search subagent: broad topic search across the corpus.

Specialized for general research questions. Calls `search_corpus` to gather
evidence, returns a structured `ResearchResult` with citations.
"""
from __future__ import annotations

from functools import lru_cache
import asyncio
from langchain.agents import create_agent

from src.agents.schemas import ResearchResult
from src.agents.mcp_tools import get_mcp_tool
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_chat_anthropic
from src.llm.prompts import SEARCH_AGENT_SYSTEM_PROMPT
from src.observability import get_callback_handler

@lru_cache(maxsize=1)
def build_search_agent():
    """Construct the search agent: Claude + search_corpus tool + structured output."""
    llm = get_chat_anthropic(SUBAGENT_MODEL)
    return create_agent(
        model=llm,
        tools=[get_mcp_tool("search_corpus")],
        system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
        response_format=ResearchResult,
    )


def run_search_agent(question: str) -> ResearchResult:
    """Run the search agent on one focused question; return ResearchResult."""
    agent = build_search_agent()
    handler = get_callback_handler()

    output =asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Research question:\n{question}"}]},
        config={"callbacks": [handler], "run_name": "search_agent"},
    ))
    return output["structured_response"]
