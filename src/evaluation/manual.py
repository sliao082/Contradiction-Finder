"""Manual labeling export and metric helpers for generated conflict cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MANUAL_EVALUATION_DIR, ensure_data_dirs
from src.evaluation.schemas import ManualLabel
from src.schemas import ConflictCard
from src.utils.io import read_jsonl, write_json

MANUAL_LABELS: list[ManualLabel] = [
    "true_contradiction",
    "partial_or_contextual_contradiction",
    "not_contradiction",
    "unclear",
]

MANUAL_COLUMNS = [
    "card_id",
    "topic_title",
    "contradiction_score",
    "possible_reason_category",
    "claim_a_text",
    "claim_b_text",
    "paper_a_title",
    "paper_b_title",
    "evidence_a",
    "evidence_b",
    "explanation",
    "manual_label",
    "notes",
]


def cards_to_manual_label_dataframe(cards: list[ConflictCard | dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in cards:
        card = item if isinstance(item, ConflictCard) else ConflictCard(**item)
        rows.append(
            {
                "card_id": card.card_id,
                "topic_title": card.topic_title,
                "contradiction_score": card.contradiction_score,
                "possible_reason_category": card.possible_reason_category,
                "claim_a_text": card.claim_a_text,
                "claim_b_text": card.claim_b_text,
                "paper_a_title": card.paper_a_title,
                "paper_b_title": card.paper_b_title,
                "evidence_a": card.evidence_a,
                "evidence_b": card.evidence_b,
                "explanation": card.explanation,
                "manual_label": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=MANUAL_COLUMNS)


def cards_to_manual_csv_bytes(cards: list[ConflictCard | dict[str, Any]]) -> bytes:
    return cards_to_manual_label_dataframe(cards).to_csv(index=False).encode("utf-8")


def resolve_cards_path(run_dir_or_cards_path: str | Path) -> Path:
    path = Path(run_dir_or_cards_path)
    if path.is_dir():
        path = path / "cards.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find conflict cards at {path}")
    return path


def export_conflict_cards_for_labeling(
    run_dir_or_cards_path: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    ensure_data_dirs()
    cards_path = resolve_cards_path(run_dir_or_cards_path)
    cards = read_jsonl(cards_path)
    if output_csv is None:
        output_csv = MANUAL_EVALUATION_DIR / f"{cards_path.parent.name}_manual_labels.csv"
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards_to_manual_label_dataframe(cards).to_csv(output_path, index=False)
    return output_path


def compute_manual_metrics(label_csv: str | Path) -> dict[str, Any]:
    df = pd.read_csv(label_csv).fillna("")
    if "manual_label" not in df.columns:
        raise ValueError("CSV must contain a manual_label column.")

    labels = df["manual_label"].astype(str).str.strip()
    valid = labels.isin(MANUAL_LABELS)
    labeled = df[valid].copy()
    labeled["manual_label"] = labels[valid].values
    label_counts = {
        label: int((labels == label).sum())
        for label in MANUAL_LABELS
    }
    non_unclear = labeled[labeled["manual_label"] != "unclear"]

    strict_positive = labeled["manual_label"] == "true_contradiction"
    broad_positive = labeled["manual_label"].isin(
        ["true_contradiction", "partial_or_contextual_contradiction"]
    )

    strict_non_unclear_positive = non_unclear["manual_label"] == "true_contradiction"
    broad_non_unclear_positive = non_unclear["manual_label"].isin(
        ["true_contradiction", "partial_or_contextual_contradiction"]
    )

    return {
        "rows": int(len(df)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(len(df) - len(labeled)),
        "label_counts": label_counts,
        "strict_precision_all_labeled": (
            round(float(strict_positive.mean()), 4) if len(labeled) else None
        ),
        "broad_precision_all_labeled": (
            round(float(broad_positive.mean()), 4) if len(labeled) else None
        ),
        "strict_precision_excluding_unclear": (
            round(float(strict_non_unclear_positive.mean()), 4)
            if len(non_unclear)
            else None
        ),
        "broad_precision_excluding_unclear": (
            round(float(broad_non_unclear_positive.mean()), 4)
            if len(non_unclear)
            else None
        ),
        "unclear_rate": (
            round(label_counts["unclear"] / len(labeled), 4) if len(labeled) else None
        ),
        "manual_label_options": MANUAL_LABELS,
        "note": (
            "Manual export contains only system-predicted conflict cards, so these metrics "
            "estimate precision/acceptance rate rather than recall."
        ),
    }


def save_manual_metrics(
    label_csv: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    ensure_data_dirs()
    metrics = compute_manual_metrics(label_csv)
    if output_path is None:
        source = Path(label_csv)
        output_path = source.with_name(f"{source.stem}_metrics.json")
    return write_json(Path(output_path), metrics)
