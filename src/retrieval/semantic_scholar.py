"""Semantic Scholar paper search with local JSONL caching."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from src.config import MIN_ABSTRACT_LENGTH, RAW_SEMANTIC_SCHOLAR_DIR, SEMANTIC_SCHOLAR_API_KEY
from src.schemas import Paper
from src.utils.hashing import query_hash
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import clean_text

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,authors,venue,url,citationCount"


class SemanticScholarError(RuntimeError):
    """Raised when retrieval fails and no cache fallback exists."""


def cache_path_for_query(query: str, limit: int) -> Path:
    return RAW_SEMANTIC_SCHOLAR_DIR / f"{query_hash(query, limit)}.jsonl"


def _paper_from_api(row: dict[str, Any], query: str) -> Paper | None:
    abstract = clean_text(row.get("abstract"))
    title = clean_text(row.get("title")) or "Untitled paper"
    paper_id = str(row.get("paperId") or "")
    if not paper_id or len(abstract) < MIN_ABSTRACT_LENGTH:
        return None

    authors = row.get("authors") or []
    author_names = [
        clean_text(author.get("name"))
        for author in authors
        if isinstance(author, dict) and clean_text(author.get("name"))
    ]

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=row.get("year"),
        authors=author_names,
        venue=clean_text(row.get("venue")) or None,
        url=row.get("url"),
        citation_count=row.get("citationCount"),
        query=query,
    )


def _load_cached(cache_path: Path) -> list[Paper]:
    return [Paper(**row) for row in read_jsonl(cache_path)]


def search_papers(query: str, limit: int = 30, use_cache: bool = True) -> list[Paper]:
    """Search Semantic Scholar and cache normalized papers as JSONL."""

    query = clean_text(query)
    if not query:
        raise SemanticScholarError("Enter a research topic query before searching.")

    limit = max(1, min(int(limit), 50))
    RAW_SEMANTIC_SCHOLAR_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_query(query, limit)

    if use_cache and cache_path.exists():
        cached = _load_cached(cache_path)
        if cached:
            return cached

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    params = {"query": query, "limit": limit, "fields": FIELDS}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code == 429:
                time.sleep(2**attempt + 1)
                continue
            response.raise_for_status()
            payload = response.json()
            papers = [
                paper
                for paper in (_paper_from_api(row, query) for row in payload.get("data", []))
                if paper is not None
            ]
            write_jsonl(cache_path, papers)
            return papers
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt + 1)
        except ValueError as exc:
            last_error = exc
            break

    if cache_path.exists():
        cached = _load_cached(cache_path)
        if cached:
            return cached

    message = "Semantic Scholar retrieval failed."
    if last_error is not None:
        message = f"{message} {last_error}"
    raise SemanticScholarError(message)

