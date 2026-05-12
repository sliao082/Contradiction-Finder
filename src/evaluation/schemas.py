"""Pydantic records used by evaluation modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvalLabel = Literal["contradiction", "entailment", "neutral"]

ManualLabel = Literal[
    "true_contradiction",
    "partial_or_contextual_contradiction",
    "not_contradiction",
    "unclear",
]


class SciFactEvalPair(BaseModel):
    pair_id: str
    scifact_claim_id: int
    split: str
    claim: str
    evidence_text: str
    evidence_sentences: list[str] = Field(default_factory=list)
    evidence_sentence_indices: list[int] = Field(default_factory=list)
    evidence_doc_id: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    abstract_title: str | None = None
    gold_label: EvalLabel
    original_label: str
    grounding_source: Literal["rationale_sentences", "abstract_fallback"]
    doc_reference_ok: bool
    sentence_indices_ok: bool
    evidence_text_ok: bool

    @property
    def grounding_ok(self) -> bool:
        return self.doc_reference_ok and self.sentence_indices_ok and self.evidence_text_ok


class SciFactPrediction(BaseModel):
    pair_id: str
    gold_label: EvalLabel
    predicted_label: EvalLabel
    entailment_score: float
    contradiction_score: float
    neutral_score: float


class ThresholdMetric(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_contradictions: int
    true_contradictions: int
    evaluated_pairs: int

