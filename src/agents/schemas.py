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
