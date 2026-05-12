"""Join scored claim pairs into display-ready conflict cards."""

from __future__ import annotations

from src.cards.groq_explainer import explain_conflict
from src.config import CONTRADICTION_THRESHOLD, MAX_CONFLICT_CARDS
from src.schemas import Claim, ClaimPair, ConflictCard, ContradictionScore, Paper
from src.utils.groq_client import get_groq_client
from src.utils.hashing import stable_id


def generate_conflict_cards(
    papers: list[Paper],
    claims: list[Claim],
    pairs: list[ClaimPair],
    scores: list[ContradictionScore],
    max_cards: int = MAX_CONFLICT_CARDS,
    use_groq: bool = True,
    contradiction_threshold: float = CONTRADICTION_THRESHOLD,
) -> list[ConflictCard]:
    paper_map = {paper.paper_id: paper for paper in papers}
    claim_map = {claim.claim_id: claim for claim in claims}
    pair_map = {pair.pair_id: pair for pair in pairs}
    client = get_groq_client() if use_groq else None

    eligible_scores = [
        score
        for score in scores
        if score.contradiction_score >= contradiction_threshold
        and score.predicted_label == "contradiction"
        and score.pair_id in pair_map
    ]
    eligible_scores.sort(key=lambda item: item.contradiction_score, reverse=True)

    cards: list[ConflictCard] = []
    for score in eligible_scores[:max_cards]:
        pair = pair_map[score.pair_id]
        claim_a = claim_map.get(pair.claim_a_id)
        claim_b = claim_map.get(pair.claim_b_id)
        if claim_a is None or claim_b is None:
            continue
        paper_a = paper_map.get(claim_a.paper_id)
        paper_b = paper_map.get(claim_b.paper_id)
        if paper_a is None or paper_b is None:
            continue

        explanation = explain_conflict(
            claim_a,
            claim_b,
            paper_a,
            paper_b,
            score.contradiction_score,
            client,
        )
        cards.append(
            ConflictCard(
                card_id=stable_id("card", pair.pair_id),
                topic_title=explanation["topic_title"],
                claim_a_id=claim_a.claim_id,
                claim_b_id=claim_b.claim_id,
                claim_a_text=claim_a.normalized_claim,
                claim_b_text=claim_b.normalized_claim,
                paper_a_title=paper_a.title,
                paper_b_title=paper_b.title,
                paper_a_year=paper_a.year,
                paper_b_year=paper_b.year,
                paper_a_url=paper_a.url,
                paper_b_url=paper_b.url,
                evidence_a=claim_a.original_sentence,
                evidence_b=claim_b.original_sentence,
                contradiction_score=score.contradiction_score,
                possible_reason_category=explanation["possible_reason_category"],
                explanation=explanation["explanation"],
            )
        )
    return cards
