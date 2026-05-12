"""Serializable Pydantic models for every pipeline artifact."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "result",
    "comparison",
    "causal",
    "conclusion",
    "limitation",
    "method_effect",
    "other",
]

ExtractionMethod = Literal[
    "heuristic_llm_validated",
    "llm_recovered",
    "heuristic_only",
]

ReasonCategory = Literal[
    "different_dataset",
    "different_metric",
    "different_population",
    "different_method",
    "different_condition",
    "different_definition",
    "unclear_from_abstract",
    "other",
]


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    url: str | None = None
    citation_count: int | None = None
    query: str
    source: str = "semantic_scholar"


class PaperSentence(BaseModel):
    sentence_id: str
    paper_id: str
    sentence_index: int
    text: str


class Claim(BaseModel):
    claim_id: str
    paper_id: str
    sentence_id: str
    original_sentence: str
    normalized_claim: str
    claim_type: ClaimType
    extraction_method: ExtractionMethod
    confidence: float


class ClaimPair(BaseModel):
    pair_id: str
    claim_a_id: str
    claim_b_id: str
    paper_a_id: str
    paper_b_id: str
    claim_a: str
    claim_b: str
    similarity_score: float


class ContradictionScore(BaseModel):
    pair_id: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    predicted_label: str


class ConflictCard(BaseModel):
    card_id: str
    topic_title: str
    claim_a_id: str
    claim_b_id: str
    claim_a_text: str
    claim_b_text: str
    paper_a_title: str
    paper_b_title: str
    paper_a_year: int | None
    paper_b_year: int | None
    paper_a_url: str | None
    paper_b_url: str | None
    evidence_a: str
    evidence_b: str
    contradiction_score: float
    possible_reason_category: ReasonCategory
    explanation: str

