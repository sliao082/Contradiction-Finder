"""Deterministic IDs and path-safe query hashes."""

from __future__ import annotations

import hashlib
import re


def sha1_short(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def query_hash(query: str, paper_limit: int) -> str:
    normalized = " ".join(query.lower().split())
    return sha1_short(f"{normalized}|limit={paper_limit}", length=12)


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{sha1_short(raw, length=length)}"


def safe_slug(value: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return (slug[:max_length] or "query").strip("-")


def safe_paper_id(paper_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", paper_id)[:80]

