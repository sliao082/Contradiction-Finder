"""Text cleanup, lightweight sentence splitting, and claim normalization helpers."""

from __future__ import annotations

import re

ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "al.",
    "vs.",
    "fig.",
    "eq.",
    "dr.",
    "mr.",
    "mrs.",
    "prof.",
}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def normalize_claim_text(text: str) -> str:
    text = clean_text(text)
    text = text.strip(" ;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def split_into_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    placeholders: dict[str, str] = {}
    protected = text
    for idx, abbr in enumerate(ABBREVIATIONS):
        token = f"__ABBR_{idx}__"
        placeholders[token] = abbr
        protected = re.sub(re.escape(abbr), token, protected, flags=re.IGNORECASE)

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", protected)
    sentences = []
    for piece in pieces:
        restored = piece.strip()
        for token, abbr in placeholders.items():
            restored = restored.replace(token, abbr)
        restored = clean_text(restored)
        if restored:
            sentences.append(restored)
    return sentences

