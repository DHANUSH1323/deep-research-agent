"""Paper-summarizer subagent: deep-reads one paper and returns a PaperSummary with key contributions, methods, and results."""


from __future__ import annotations

import asyncio
from functools import lru_cache

from langchain.agents import create_agent

from src.agents.schemas import PaperSummary
from src.agents.mcp_tools import get_mcp_tool
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_chat_anthropic
from src.llm.prompts import PAPER_SUMMARIZER_SYSTEM_PROMPT
from src.observability import get_callback_handler

@lru_cache(maxsize=1)
def build_paper_summarizer():
    """Construct the paper summarizer agent: Claude + get_paper_full_text + structured output."""
    llm = get_chat_anthropic(SUBAGENT_MODEL)
    return create_agent(
        model=llm,
        tools=[get_mcp_tool("get_paper_full_text")],
        system_prompt=PAPER_SUMMARIZER_SYSTEM_PROMPT,
        response_format=PaperSummary,
    )


def run_paper_summarizer(arxiv_id: str) -> PaperSummary:
    """Run the paper summarizer agent on one paper; return PaperSummary."""
    agent = build_paper_summarizer()
    handler = get_callback_handler()

    output = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Summarize this paper:\n{arxiv_id}"}]},
        config={"callbacks": [handler], "run_name": "paper_summarizer"},
    ))
    return output["structured_response"]