"""Groq-backed claim validation, normalization, and abstract recovery."""

from __future__ import annotations

from typing import Any

from src.schemas import Claim, ClaimType, Paper, PaperSentence
from src.utils.groq_client import GroqJsonClient
from src.utils.hashing import safe_paper_id, stable_id
from src.utils.text import clean_text, normalize_claim_text

VALID_CLAIM_TYPES = {
    "result",
    "comparison",
    "causal",
    "conclusion",
    "limitation",
    "method_effect",
    "other",
}


def _claim_type(value: Any) -> ClaimType:
    value = str(value or "other").strip().lower()
    if value not in VALID_CLAIM_TYPES:
        return "other"
    return value  # type: ignore[return-value]


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _validation_prompt(sentence: PaperSentence, paper: Paper) -> str:
    return f"""You are helping extract scientific claims from research paper abstracts.
A scientific claim is a sentence that expresses a concrete finding, result, comparison, causal relationship, limitation, or conclusion.
Given the paper title and sentence below, decide whether the sentence expresses a scientific claim.
Return only valid JSON matching this schema:
{{
  "is_claim": boolean,
  "normalized_claim": string,
  "claim_type": "result" | "comparison" | "causal" | "conclusion" | "limitation" | "method_effect" | "other",
  "confidence": number,
  "reason": string
}}
Rules:
- Only use information from the given sentence.
- Do not invent details.
- The normalized claim should be concise and standalone.
- If the sentence is vague, procedural, or only describes the paper structure, return is_claim=false.
Paper title:
{paper.title}
Sentence:
{sentence.text}
"""


def validate_and_normalize_claim(
    sentence: PaperSentence,
    paper: Paper,
    client: GroqJsonClient,
) -> Claim | None:
    data = client.generate_json(_validation_prompt(sentence, paper))
    if not data.get("is_claim"):
        return None
    normalized = normalize_claim_text(str(data.get("normalized_claim") or sentence.text))
    if not normalized:
        return None
    return Claim(
        claim_id=stable_id("claim", paper.paper_id, sentence.sentence_id, normalized),
        paper_id=paper.paper_id,
        sentence_id=sentence.sentence_id,
        original_sentence=sentence.text,
        normalized_claim=normalized,
        claim_type=_claim_type(data.get("claim_type")),
        extraction_method="heuristic_llm_validated",
        confidence=_confidence(data.get("confidence"), default=0.75),
    )


def _recovery_prompt(paper: Paper, max_claims: int) -> str:
    return f"""You are extracting key scientific claims from a paper abstract.
Return up to {max_claims} concrete claim-like statements grounded only in the abstract.
Return only valid JSON:
{{
  "claims": [
    {{
      "normalized_claim": string,
      "source_sentence": string,
      "claim_type": "result" | "comparison" | "causal" | "conclusion" | "limitation" | "method_effect" | "other",
      "confidence": number
    }}
  ]
}}
Rules:
- Do not invent claims not stated in the abstract.
- Prefer findings, results, comparisons, and conclusions.
- Avoid generic background statements.
- Avoid method descriptions unless they state an effect or result.
Paper title:
{paper.title}
Abstract:
{paper.abstract}
"""


def recover_claims_from_abstract(
    paper: Paper,
    client: GroqJsonClient,
    max_claims: int = 3,
) -> list[Claim]:
    data = client.generate_json(_recovery_prompt(paper, max_claims))
    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        return []

    claims: list[Claim] = []
    safe_id = safe_paper_id(paper.paper_id)
    for idx, raw in enumerate(raw_claims[:max_claims]):
        if not isinstance(raw, dict):
            continue
        normalized = normalize_claim_text(str(raw.get("normalized_claim") or ""))
        source_sentence = clean_text(str(raw.get("source_sentence") or normalized))
        if not normalized or not source_sentence:
            continue
        sentence_id = f"{safe_id}_G{idx}"
        claims.append(
            Claim(
                claim_id=stable_id("claim", paper.paper_id, sentence_id, normalized),
                paper_id=paper.paper_id,
                sentence_id=sentence_id,
                original_sentence=source_sentence,
                normalized_claim=normalized,
                claim_type=_claim_type(raw.get("claim_type")),
                extraction_method="llm_recovered",
                confidence=_confidence(raw.get("confidence"), default=0.65),
            )
        )
    return claims
