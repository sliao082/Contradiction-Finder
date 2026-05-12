"""Claim extraction orchestration."""

from __future__ import annotations

from src.claims.groq_extractor import recover_claims_from_abstract, validate_and_normalize_claim
from src.claims.heuristic_extractor import heuristic_claim_candidates
from src.schemas import Claim, Paper, PaperSentence
from src.utils.groq_client import get_groq_client
from src.utils.hashing import stable_id
from src.utils.text import normalize_claim_text

METHOD_PRIORITY = {
    "heuristic_llm_validated": 3,
    "llm_recovered": 2,
    "heuristic_only": 1,
}


def _heuristic_only_claim(sentence: PaperSentence) -> Claim:
    normalized = normalize_claim_text(sentence.text)
    return Claim(
        claim_id=stable_id("claim", sentence.paper_id, sentence.sentence_id, normalized),
        paper_id=sentence.paper_id,
        sentence_id=sentence.sentence_id,
        original_sentence=sentence.text,
        normalized_claim=normalized,
        claim_type="other",
        extraction_method="heuristic_only",
        confidence=0.45,
    )


def _deduplicate(claims: list[Claim]) -> list[Claim]:
    best: dict[tuple[str, str], Claim] = {}
    for claim in claims:
        key = (claim.paper_id, claim.normalized_claim.lower().strip())
        current = best.get(key)
        if current is None:
            best[key] = claim
            continue
        current_priority = METHOD_PRIORITY.get(current.extraction_method, 0)
        new_priority = METHOD_PRIORITY.get(claim.extraction_method, 0)
        if (new_priority, claim.confidence) > (current_priority, current.confidence):
            best[key] = claim
    return sorted(best.values(), key=lambda item: (item.paper_id, item.claim_id))


def extract_claims(
    papers: list[Paper],
    sentences: list[PaperSentence],
    use_groq: bool = True,
    recover_missing: bool = True,
) -> list[Claim]:
    paper_map = {paper.paper_id: paper for paper in papers}
    candidates = heuristic_claim_candidates(sentences)
    claims: list[Claim] = []

    client = get_groq_client() if use_groq else None
    if client is None:
        claims.extend(_heuristic_only_claim(sentence) for sentence in candidates)
    else:
        for sentence in candidates:
            paper = paper_map.get(sentence.paper_id)
            if paper is None:
                continue
            try:
                claim = validate_and_normalize_claim(sentence, paper, client)
            except Exception:
                claim = _heuristic_only_claim(sentence)
            if claim is not None:
                claims.append(claim)

        if recover_missing:
            for paper in papers:
                try:
                    claims.extend(recover_claims_from_abstract(paper, client, max_claims=3))
                except Exception:
                    continue

    return _deduplicate(claims)
