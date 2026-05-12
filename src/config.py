"""Application configuration and default runtime paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_SEMANTIC_SCHOLAR_DIR = DATA_DIR / "raw" / "semantic_scholar"
PROCESSED_RUNS_DIR = DATA_DIR / "processed" / "runs"
CACHE_DIR = DATA_DIR / "cache"
EVALUATION_DIR = DATA_DIR / "evaluation"
SCIFACT_EVALUATION_DIR = EVALUATION_DIR / "scifact"
MANUAL_EVALUATION_DIR = EVALUATION_DIR / "manual"

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
).strip()

DEFAULT_PAPER_LIMIT = int(os.getenv("DEFAULT_PAPER_LIMIT", "30"))
MIN_ABSTRACT_LENGTH = int(os.getenv("MIN_ABSTRACT_LENGTH", "300"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
MAX_PAIRS_PER_CLAIM = int(os.getenv("MAX_PAIRS_PER_CLAIM", "5"))
CONTRADICTION_THRESHOLD = float(os.getenv("CONTRADICTION_THRESHOLD", "0.65"))
MAX_CONFLICT_CARDS = int(os.getenv("MAX_CONFLICT_CARDS", "20"))


def ensure_data_dirs() -> None:
    """Create all data directories used by the pipeline."""

    for path in (
        RAW_SEMANTIC_SCHOLAR_DIR,
        PROCESSED_RUNS_DIR,
        CACHE_DIR,
        SCIFACT_EVALUATION_DIR,
        MANUAL_EVALUATION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
