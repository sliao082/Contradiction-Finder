"""Build related cross-paper claim pairs by cosine similarity."""

from __future__ import annotations

import numpy as np

from src.config import MAX_PAIRS_PER_CLAIM
from src.schemas import Claim, ClaimPair
from src.utils.hashing import stable_id


def build_candidate_pairs(
    claims: list[Claim],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.45,
    max_pairs_per_claim: int = MAX_PAIRS_PER_CLAIM,
) -> list[ClaimPair]:
    if len(claims) < 2 or embeddings.size == 0:
        return []
    if embeddings.shape[0] != len(claims):
        raise ValueError("Embedding count does not match claim count.")

    similarities = embeddings @ embeddings.T
    pairs: list[ClaimPair] = []
    seen: set[tuple[str, str]] = set()

    for i, claim in enumerate(claims):
        ranked = np.argsort(similarities[i])[::-1]
        kept_for_claim = 0
        for j in ranked:
            if i == j:
                continue
            other = claims[int(j)]
            if other.paper_id == claim.paper_id:
                continue
            similarity = float(similarities[i, j])
            if similarity < similarity_threshold:
                break
            pair_key = tuple(sorted([claim.claim_id, other.claim_id]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            left, right = (claim, other) if claim.claim_id <= other.claim_id else (other, claim)
            pairs.append(
                ClaimPair(
                    pair_id=stable_id("pair", left.claim_id, right.claim_id),
                    claim_a_id=left.claim_id,
                    claim_b_id=right.claim_id,
                    paper_a_id=left.paper_id,
                    paper_b_id=right.paper_id,
                    claim_a=left.normalized_claim,
                    claim_b=right.normalized_claim,
                    similarity_score=round(similarity, 4),
                )
            )
            kept_for_claim += 1
            if kept_for_claim >= max_pairs_per_claim:
                break

    return sorted(pairs, key=lambda pair: pair.similarity_score, reverse=True)

