"""Small Groq API wrapper for JSON-only model calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL


class GroqUnavailable(RuntimeError):
    """Raised when Groq cannot be used for the current run."""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Groq returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass(frozen=True)
class GroqJsonClient:
    model_name: str = GROQ_MODEL
    api_key: str = GROQ_API_KEY
    api_url: str = GROQ_API_URL

    def __post_init__(self) -> None:
        if not self.api_key:
            raise GroqUnavailable("GROQ_API_KEY is not set.")

    @retry(
        retry=retry_if_exception_type((ValueError, RuntimeError, json.JSONDecodeError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate_json(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise GroqUnavailable(
                f"Groq API request failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Groq returned an unexpected response shape.") from exc
        return extract_json(text)


def get_groq_client(model_name: str | None = None) -> GroqJsonClient | None:
    try:
        return GroqJsonClient(model_name=model_name or GROQ_MODEL)
    except GroqUnavailable:
        return None
