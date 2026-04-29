"""Pydantic schemas for agent inputs and outputs."""
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One reference to a chunk in the corpus."""
    chunk_id: str = Field(description="Unique identifier for the source chunk.")
    arxiv_id: str = Field(description="ArXiv identifier of the source paper.")
    page_num: int = Field(description="Page number in the source paper.")


class Finding(BaseModel):
    """One factual claim with supporting citations."""
    claim: str = Field(description="A specific factual statement extracted from the corpus.")
    citations: list[Citation] = Field(description="Citations that support the claim. At least one required.")


class ResearchResult(BaseModel):
    """Output of one research subagent for one sub-question."""
    sub_question: str = Field(description="The sub-question this result addresses.")
    findings: list[Finding] = Field(description="The cited findings gathered.")
    summary: str = Field(description="A 2-3 sentence synthesis of the findings.")

class PaperSummary(BaseModel):
    """Summary of one paper, including its contributions and methods."""
    arxiv_id: str = Field(description="Arxiv identifier of the paper.")
    summary: str = Field(description="A 2-3 sentence summary of this paper for non-exports")
    key_contributions: list[Finding] = Field(description="The 2-4 main contributins of paper claims, each with citations to the chunks where they're supported.")
    methodology: list[Finding] = Field(description="A description of the methods used in the paper, with citations to the chunks where they're described.")
    notable_results: list[Finding] = Field(description="A description of the main results of the paper, with citations to the chunks where they're described.")

class FinalReport(BaseModel):
    """Final synthesized answer to a user research query, aggregated from subagent outputs."""
    user_query: str = Field(description="The original research question from the user.")
    executive_summary: str = Field(description="A 3-5 sentence direct answer to the user's question, synthesized from subagent findings. Every claim must be traceable to citations in findings.")
    findings: list[Finding] = Field(description="Aggregated cited findings from subagent outputs that support the executive summary. At least one required.")