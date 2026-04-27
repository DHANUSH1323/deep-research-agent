"""Search subagent: broad topic search across the corpus.

Specialized for general research questions. Calls `search_corpus` to gather
evidence, returns a structured `ResearchResult` with citations.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

from src.agents.schemas import ResearchResult
from src.agents.tools import search_corpus
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_chat_anthropic
from src.llm.prompts import SEARCH_AGENT_SYSTEM_PROMPT


def build_search_agent():
    """Construct the search agent: Claude + search_corpus tool + structured output."""
    llm = get_chat_anthropic(SUBAGENT_MODEL)
    return create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
        response_format=ResearchResult,
    )


def run_search_agent(question: str) -> ResearchResult:
    """Run the search agent on one focused question; return ResearchResult."""
    agent = build_search_agent()
    handler = CallbackHandler()

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"Research question:\n{question}"}]},
        config={"callbacks": [handler]},
    )
    return output["structured_response"]
