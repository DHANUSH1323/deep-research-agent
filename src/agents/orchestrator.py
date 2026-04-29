"""Orchestrator: Sonnet 4.6 agent that dispatches specialists and synthesizes a cited FinalReport."""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.agents.paper_summarizer import run_paper_summarizer
from src.agents.schemas import FinalReport
from src.agents.search_agent import run_search_agent
from src.config import ORCHESTRATOR_MODEL
from src.llm.anthropic_client import get_chat_anthropic
from src.llm.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from src.observability import get_callback_handler


@tool
def dispatch_search_agent(sub_question: str) -> str:
    """Dispatch the search agent for a sub-question. Returns the ResearchResult JSON."""
    result = run_search_agent(sub_question)
    return result.model_dump_json(indent=2)


@tool
def dispatch_paper_summarizer(arxiv_id: str) -> str:
    """Dispatch the paper summarizer for an arxiv_id. Returns the PaperSummary JSON."""
    result = run_paper_summarizer(arxiv_id)
    return result.model_dump_json(indent=2)


@lru_cache(maxsize=1)
def build_orchestrator():
    """Construct the orchestrator agent: Claude + dispatch tools + structured output."""
    llm = get_chat_anthropic(ORCHESTRATOR_MODEL)
    return create_agent(
        model=llm,
        tools=[dispatch_search_agent, dispatch_paper_summarizer],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        response_format=FinalReport,
    )


def run_orchestrator(user_query: str) -> FinalReport:
    """Run the orchestrator agent on a user query; return the FinalReport."""
    agent = build_orchestrator()
    handler = get_callback_handler()

    output = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"callbacks": [handler]},
    )
    return output["structured_response"]