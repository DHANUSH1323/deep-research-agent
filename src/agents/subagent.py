"""Research subagent built on LangChain's prebuilt ReAct agent.

The agent calls `search_corpus` as needed and returns a structured `ResearchResult`.
Tracing is automatic via Langfuse's LangChain callback handler.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

from src.agents.schemas import ResearchResult
from src.agents.tools import search_corpus
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_chat_anthropic
from src.llm.prompts import SUBAGENT_SYSTEM_PROMPT


def build_subagent():
    """Construct the subagent: Claude + tools + structured output."""
    llm = get_chat_anthropic(SUBAGENT_MODEL)
    return create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        response_format=ResearchResult,
    )


def research(sub_question: str) -> ResearchResult:
    """Run the subagent on one sub-question; return a structured ResearchResult."""
    agent = build_subagent()
    handler = CallbackHandler()

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"Sub-question:\n{sub_question}"}]},
        config={"callbacks": [handler]},
    )
    return output["structured_response"]
