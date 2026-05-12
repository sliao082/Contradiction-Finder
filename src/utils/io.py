"""JSON and JSONL persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

try:
    import orjson
except ImportError:  # pragma: no cover - optional speedup
    orjson = None


def model_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        return item.dict()
    if isinstance(item, dict):
        return item
    raise TypeError(f"Cannot serialize item of type {type(item)!r}")


def dumps_json(value: Any) -> str:
    if orjson is not None:
        return orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8")
    return json.dumps(value, ensure_ascii=False) + "\n"


def write_jsonl(path: Path, items: Iterable[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(dumps_json(model_to_dict(item)))
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_bytes_if_exists(path: str | Path) -> bytes:
    resolved = Path(path)
    return resolved.read_bytes() if resolved.exists() else b""

