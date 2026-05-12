"""SentenceTransformer claim embeddings."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.schemas import Claim


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements.txt."
        ) from exc
    return SentenceTransformer(model_name)


class ClaimEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_sentence_transformer(model_name)

    def encode_claims(self, claims: list[Claim]) -> np.ndarray:
        if not claims:
            return np.empty((0, 0), dtype=np.float32)
        texts = [claim.normalized_claim for claim in claims]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

