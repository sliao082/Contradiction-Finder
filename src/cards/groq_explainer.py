"""Groq-backed cautious explanations for possible contradictions."""

from __future__ import annotations

from typing import Any

from src.schemas import Claim, Paper, ReasonCategory
from src.utils.groq_client import GroqJsonClient
from src.utils.text import clean_text

VALID_REASON_CATEGORIES = {
    "different_dataset",
    "different_metric",
    "different_population",
    "different_method",
    "different_condition",
    "different_definition",
    "unclear_from_abstract",
    "other",
}


def fallback_explanation(
    claim_a: Claim,
    claim_b: Claim,
    paper_a: Paper,
    paper_b: Paper,
) -> dict[str, str]:
    return {
        "topic_title": "Claims appear to differ across abstracts",
        "possible_reason_category": "unclear_from_abstract",
        "explanation": (
            "These claims may be in tension, but the abstracts alone do not provide "
            "enough context to identify whether the difference comes from data, methods, "
            "metrics, or conditions."
        ),
    }


def _reason_category(value: Any) -> ReasonCategory:
    value = str(value or "unclear_from_abstract").strip().lower()
    if value not in VALID_REASON_CATEGORIES:
        return "unclear_from_abstract"
    return value  # type: ignore[return-value]


def _prompt(
    claim_a: Claim,
    claim_b: Claim,
    paper_a: Paper,
    paper_b: Paper,
    contradiction_score: float,
) -> str:
    return f"""You are helping a researcher inspect a possible contradiction between two scientific claims.
Use only the provided claims and evidence snippets. Do not invent missing details.
Return only valid JSON:
{{
  "topic_title": string,
  "possible_reason_category": "different_dataset" | "different_metric" | "different_population" | "different_method" | "different_condition" | "different_definition" | "unclear_from_abstract" | "other",
  "explanation": string
}}
Rules:
- The title should describe the disagreement in 5-12 words.
- The explanation should be 1-2 sentences.
- If the reason is not clear from the evidence, choose "unclear_from_abstract".
- Use cautious language such as "may", "appears", or "could".
- Do not state that one paper is correct and the other is wrong.
Paper A title:
{paper_a.title}
Claim A:
{claim_a.normalized_claim}
Evidence A:
{claim_a.original_sentence}
Paper B title:
{paper_b.title}
Claim B:
{claim_b.normalized_claim}
Evidence B:
{claim_b.original_sentence}
NLI contradiction score:
{contradiction_score:.4f}
"""


def explain_conflict(
    claim_a: Claim,
    claim_b: Claim,
    paper_a: Paper,
    paper_b: Paper,
    contradiction_score: float,
    client: GroqJsonClient | None,
) -> dict[str, str]:
    if client is None:
        return fallback_explanation(claim_a, claim_b, paper_a, paper_b)
    try:
        data = client.generate_json(
            _prompt(claim_a, claim_b, paper_a, paper_b, contradiction_score)
        )
    except Exception:
        return fallback_explanation(claim_a, claim_b, paper_a, paper_b)

    title = clean_text(str(data.get("topic_title") or "Claims appear to differ"))
    explanation = clean_text(str(data.get("explanation") or ""))
    if not explanation:
        return fallback_explanation(claim_a, claim_b, paper_a, paper_b)
    return {
        "topic_title": title[:140],
        "possible_reason_category": _reason_category(data.get("possible_reason_category")),
        "explanation": explanation,
    }
