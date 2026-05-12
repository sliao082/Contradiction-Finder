"""Transparent claim-like sentence heuristics."""

from __future__ import annotations

import re

from src.schemas import PaperSentence
from src.utils.text import word_count

CLAIM_PATTERNS = [
    "we find",
    "we found",
    "we show",
    "we demonstrate",
    "our results show",
    "results show",
    "we observe",
    "we conclude",
    "suggests that",
    "indicates that",
    "significantly",
    "improves",
    "improved",
    "reduces",
    "reduced",
    "increases",
    "increased",
    "outperforms",
    "underperforms",
    "associated with",
    "correlated with",
    "leads to",
    "causes",
    "does not improve",
    "no significant",
    "fails to",
    "less effective",
    "more effective",
    "lower than",
    "higher than",
]

STRUCTURE_PATTERNS = [
    r"^in this (paper|work|study),? we (present|propose|introduce|describe)\b",
    r"^this (paper|work|study) (presents|proposes|introduces|describes)\b",
]

EFFECT_TERMS = {
    "improves",
    "reduces",
    "increases",
    "outperforms",
    "underperforms",
    "significant",
    "effective",
    "fails",
    "leads",
}


def _looks_structural_only(text: str) -> bool:
    lowered = text.lower()
    if not any(re.search(pattern, lowered) for pattern in STRUCTURE_PATTERNS):
        return False
    return not any(term in lowered for term in EFFECT_TERMS)


def heuristic_claim_candidates(sentences: list[PaperSentence]) -> list[PaperSentence]:
    candidates: list[PaperSentence] = []
    seen: set[str] = set()
    for sentence in sentences:
        text = sentence.text.strip()
        lowered = text.lower()
        if sentence.sentence_id in seen or word_count(text) < 8:
            continue
        if _looks_structural_only(text):
            continue
        if any(pattern in lowered for pattern in CLAIM_PATTERNS):
            candidates.append(sentence)
            seen.add(sentence.sentence_id)
    return candidates

