"""Paper_summarizer: This agent is responsible for depp reading one paper and 
returns a PaperSummary with key contributions, methods and results"""

from __future__ import annotations

from src import config
from src.agents.schemas import PaperSummary
from src.agents.tools import get_paper_full_text
from src.llm.prompts import PAPER_SUMMARIZER_SYSTEM_PROMPT

from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from src.config import SUBAGENT_MODEL
from src.llm.anthropic_client import get_chat_anthropic

def build_paper_summarizer():
    """Construct the paper summarizer agent: Claude + get_paper_full_text + structured output."""
    llm = get_chat_anthropic(SUBAGENT_MODEL)
    return create_agent(
        model=llm,
        tools=[get_paper_full_text],
        system_prompt=PAPER_SUMMARIZER_SYSTEM_PROMPT,
        response_format=PaperSummary,
    )


def run_paper_summarizer(arxiv_id: str) -> PaperSummary:
    """Run the paper summarizer agent on one paper; return PaperSummary."""
    agent = build_paper_summarizer()
    handler = CallbackHandler()

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"Summarize this paper:\n{arxiv_id}"}]},
        config={"callbacks": [handler]},
    )
    return output["structured_response"]