"""End-to-end contradiction finder pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cards.card_generator import generate_conflict_cards
from src.claims.claim_pipeline import extract_claims
from src.clustering.pair_builder import build_candidate_pairs
from src.config import (
    CONTRADICTION_THRESHOLD,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MAX_CONFLICT_CARDS,
    MAX_PAIRS_PER_CLAIM,
    NLI_MODEL,
    PROCESSED_RUNS_DIR,
    SEMANTIC_SCHOLAR_API_KEY,
    SIMILARITY_THRESHOLD,
    ensure_data_dirs,
)
from src.embeddings.embedder import ClaimEmbedder
from src.nli.contradiction_scorer import ContradictionScorer
from src.preprocessing.sentence_splitter import split_papers_into_sentences
from src.retrieval.semantic_scholar import search_papers
from src.schemas import ClaimPair, ConflictCard, ContradictionScore
from src.utils.hashing import query_hash
from src.utils.io import write_json, write_jsonl
from src.utils.text import clean_text


def _run_dir(query: str, paper_limit: int) -> Path:
    return PROCESSED_RUNS_DIR / query_hash(query, paper_limit)


def _paths(run_dir: Path) -> dict[str, Path]:
    return {
        "papers": run_dir / "papers.jsonl",
        "sentences": run_dir / "sentences.jsonl",
        "claims": run_dir / "claims.jsonl",
        "pairs": run_dir / "pairs.jsonl",
        "scores": run_dir / "scores.jsonl",
        "cards": run_dir / "cards.jsonl",
        "stats": run_dir / "stats.json",
    }


def _stats(
    query: str,
    paper_limit: int,
    run_dir: Path,
    warnings: list[str],
    artifacts: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    papers = artifacts.get("papers", [])
    sentences = artifacts.get("sentences", [])
    claims = artifacts.get("claims", [])
    pairs = artifacts.get("pairs", [])
    scores = artifacts.get("scores", [])
    cards = artifacts.get("cards", [])
    possible_contradictions = [
        score
        for score in scores
        if score.predicted_label == "contradiction"
        and score.contradiction_score >= settings["contradiction_threshold"]
    ]
    return {
        "query": query,
        "paper_limit": paper_limit,
        "run_dir": str(run_dir),
        "settings": settings,
        "counts": {
            "papers": len(papers),
            "sentences": len(sentences),
            "claims": len(claims),
            "candidate_pairs": len(pairs),
            "scored_pairs": len(scores),
            "possible_contradictions": len(possible_contradictions),
            "conflict_cards": len(cards),
        },
        "warnings": warnings,
    }


def run_contradiction_pipeline(
    query: str,
    paper_limit: int = 30,
    use_cache: bool = True,
    use_groq: bool = True,
    recover_missing_claims: bool = True,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    contradiction_threshold: float = CONTRADICTION_THRESHOLD,
    max_cards: int = MAX_CONFLICT_CARDS,
) -> dict[str, Any]:
    """Run the full pipeline and save every intermediate artifact to JSONL."""

    ensure_data_dirs()
    query = clean_text(query)
    if not query:
        raise ValueError("Enter a research topic query.")

    paper_limit = max(10, min(int(paper_limit), 50))
    max_cards = max(1, int(max_cards))
    run_dir = _run_dir(query, paper_limit)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(run_dir)
    warnings: list[str] = []
    if not SEMANTIC_SCHOLAR_API_KEY:
        warnings.append(
            "SEMANTIC_SCHOLAR_API_KEY is not set, so retrieval will use unauthenticated API access."
        )
    if use_groq and not GROQ_API_KEY:
        warnings.append(
            "GROQ_API_KEY is not set, so Groq validation, recovery, and explanations will use fallbacks."
        )

    settings = {
        "use_cache": use_cache,
        "use_groq": use_groq,
        "recover_missing_claims": recover_missing_claims,
        "similarity_threshold": similarity_threshold,
        "contradiction_threshold": contradiction_threshold,
        "max_cards": max_cards,
        "embedding_model": EMBEDDING_MODEL,
        "nli_model": NLI_MODEL,
        "max_pairs_per_claim": MAX_PAIRS_PER_CLAIM,
    }

    papers = search_papers(query, limit=paper_limit, use_cache=use_cache)
    write_jsonl(paths["papers"], papers)
    if not papers:
        warnings.append("No papers with sufficiently long abstracts were found.")

    sentences = split_papers_into_sentences(papers)
    write_jsonl(paths["sentences"], sentences)
    if not sentences and papers:
        warnings.append("No usable abstract sentences were found.")

    claims = extract_claims(
        papers,
        sentences,
        use_groq=use_groq,
        recover_missing=recover_missing_claims,
    )
    write_jsonl(paths["claims"], claims)
    if not claims and papers:
        warnings.append(
            "No claims were extracted. Try lowering constraints, enabling Groq recovery, or changing the query."
        )

    pairs: list[ClaimPair] = []
    scores: list[ContradictionScore] = []
    cards: list[ConflictCard] = []

    if len(claims) >= 2:
        try:
            embeddings = ClaimEmbedder(EMBEDDING_MODEL).encode_claims(claims)
            pairs = build_candidate_pairs(
                claims,
                embeddings,
                similarity_threshold=similarity_threshold,
                max_pairs_per_claim=MAX_PAIRS_PER_CLAIM,
            )
        except Exception as exc:
            warnings.append(f"Embedding or pair building failed: {exc}")
    write_jsonl(paths["pairs"], pairs)

    if pairs:
        try:
            scores = ContradictionScorer(NLI_MODEL).score_pairs(pairs)
        except Exception as exc:
            warnings.append(
                "Local NLI scoring failed. Install dependencies and allow Hugging Face model downloads. "
                f"Details: {exc}"
            )
    else:
        if claims:
            warnings.append(
                "No candidate claim pairs met the similarity threshold or cross-paper constraint."
            )
    write_jsonl(paths["scores"], scores)

    if scores:
        cards = generate_conflict_cards(
            papers,
            claims,
            pairs,
            scores,
            max_cards=max_cards,
            use_groq=use_groq,
            contradiction_threshold=contradiction_threshold,
        )
        if not cards:
            warnings.append(
                "No high-confidence possible contradictions met the selected threshold."
            )
    write_jsonl(paths["cards"], cards)

    artifacts = {
        "papers": papers,
        "sentences": sentences,
        "claims": claims,
        "pairs": pairs,
        "scores": scores,
        "cards": cards,
    }
    stats = _stats(query, paper_limit, run_dir, warnings, artifacts, settings)
    write_json(paths["stats"], stats)

    return {
        **artifacts,
        "stats": stats,
        "paths": {key: str(path) for key, path in paths.items()},
        "run_dir": str(run_dir),
        "warnings": warnings,
    }
