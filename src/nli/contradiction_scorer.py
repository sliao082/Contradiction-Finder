"""Local Hugging Face NLI scorer."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.schemas import ClaimPair, ContradictionScore


@lru_cache(maxsize=2)
def _load_nli_components(model_name: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "transformers and torch are required for local NLI scoring."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def _label_indices(model) -> dict[str, int]:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    indices: dict[str, int] = {}
    for idx, label in id2label.items():
        if "contrad" in label:
            indices["contradiction"] = idx
        elif "entail" in label:
            indices["entailment"] = idx
        elif "neutral" in label:
            indices["neutral"] = idx

    label2id = {str(k).lower(): int(v) for k, v in getattr(model.config, "label2id", {}).items()}
    for key in ("contradiction", "entailment", "neutral"):
        if key not in indices and key in label2id:
            indices[key] = label2id[key]

    if len(id2label) == 3:
        fallback = {"contradiction": 0, "entailment": 1, "neutral": 2}
        for key, idx in fallback.items():
            indices.setdefault(key, idx)

    for key in ("contradiction", "entailment", "neutral"):
        indices.setdefault(key, 0)
    return indices


class ContradictionScorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer, self.model, self.torch = _load_nli_components(model_name)
        self.indices = _label_indices(self.model)

    def _predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return {
            "entailment": float(probs[self.indices["entailment"]]),
            "contradiction": float(probs[self.indices["contradiction"]]),
            "neutral": float(probs[self.indices["neutral"]]),
        }

    def score_pair(self, pair: ClaimPair) -> ContradictionScore:
        forward = self._predict(pair.claim_a, pair.claim_b)
        reverse = self._predict(pair.claim_b, pair.claim_a)
        combined = {
            label: max(forward[label], reverse[label])
            for label in ("entailment", "contradiction", "neutral")
        }
        predicted_label = max(combined, key=combined.get)
        return ContradictionScore(
            pair_id=pair.pair_id,
            entailment_score=round(combined["entailment"], 4),
            contradiction_score=round(combined["contradiction"], 4),
            neutral_score=round(combined["neutral"], 4),
            predicted_label=predicted_label,
        )

    def score_pairs(self, pairs: list[ClaimPair]) -> list[ContradictionScore]:
        scores: list[ContradictionScore] = []
        for pair in pairs:
            try:
                scores.append(self.score_pair(pair))
            except Exception:
                scores.append(
                    ContradictionScore(
                        pair_id=pair.pair_id,
                        entailment_score=0.0,
                        contradiction_score=0.0,
                        neutral_score=1.0,
                        predicted_label="neutral",
                    )
                )
        return scores

