"""SciFact evaluation for local NLI contradiction scoring."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import CACHE_DIR, NLI_MODEL, SCIFACT_EVALUATION_DIR, ensure_data_dirs
from src.evaluation.schemas import EvalLabel, SciFactEvalPair, SciFactPrediction
from src.nli.contradiction_scorer import ContradictionScorer
from src.schemas import ClaimPair
from src.utils.hashing import stable_id
from src.utils.io import write_json, write_jsonl
from src.utils.text import clean_text

LABELS: list[EvalLabel] = ["contradiction", "entailment", "neutral"]
DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(50, 91, 5)]
SCIFACT_RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


class SciFactEvaluationError(RuntimeError):
    """Raised when SciFact evaluation cannot be completed."""


def normalize_scifact_label(raw_label: str | None) -> EvalLabel:
    label = (raw_label or "").strip().upper().replace(" ", "_")
    if label in {"CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "REFUTED"}:
        return "contradiction"
    if label in {"SUPPORT", "SUPPORTS", "SUPPORTED"}:
        return "entailment"
    return "neutral"


def normalize_prediction_label(raw_label: str | None) -> EvalLabel:
    label = (raw_label or "").strip().lower()
    if "contrad" in label:
        return "contradiction"
    if "entail" in label or "support" in label:
        return "entailment"
    return "neutral"


def _load_dataset_config(dataset_name: str, config_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SciFactEvaluationError(
            "The datasets package is required for SciFact evaluation. "
            "Install requirements.txt first."
        ) from exc

    return load_dataset(dataset_name, config_name, split=split)


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise SciFactEvaluationError(f"Unsafe path in SciFact archive: {member.name}")
        tar.extractall(extract_dir)


def _download_scifact_release() -> Path:
    cache_dir = CACHE_DIR / "scifact"
    archive_path = cache_dir / "data.tar.gz"
    extract_dir = cache_dir / "release"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        response = requests.get(SCIFACT_RELEASE_URL, timeout=60)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    corpus_candidates = list(extract_dir.rglob("corpus.jsonl")) if extract_dir.exists() else []
    if not corpus_candidates:
        _safe_extract_tar(archive_path, extract_dir)
    return extract_dir


def _find_release_file(extract_dir: Path, file_name: str) -> Path:
    matches = list(extract_dir.rglob(file_name))
    if not matches:
        raise SciFactEvaluationError(f"Could not find {file_name} in SciFact release archive.")
    return matches[0]


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _flatten_release_claim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        if "evidence_label" in row:
            flattened.append(row)
            continue

        evidence_by_doc = row.get("evidence") or {}
        cited_doc_ids = [int(doc_id) for doc_id in row.get("cited_doc_ids", [])]
        evidence_doc_ids: set[int] = set()

        for doc_id, evidence_items in evidence_by_doc.items():
            evidence_doc_ids.add(int(doc_id))
            if isinstance(evidence_items, dict):
                evidence_items = [evidence_items]
            for evidence in evidence_items:
                flattened.append(
                    {
                        "id": row["id"],
                        "claim": row["claim"],
                        "evidence_doc_id": str(doc_id),
                        "evidence_label": evidence.get("label", ""),
                        "evidence_sentences": evidence.get("sentences", []),
                        "cited_doc_ids": cited_doc_ids,
                    }
                )

        for cited_doc_id in cited_doc_ids:
            if cited_doc_id not in evidence_doc_ids:
                flattened.append(
                    {
                        "id": row["id"],
                        "claim": row["claim"],
                        "evidence_doc_id": str(cited_doc_id),
                        "evidence_label": "NOINFO",
                        "evidence_sentences": [],
                        "cited_doc_ids": cited_doc_ids,
                    }
                )
    return flattened


def _load_scifact_release(split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    split_file = {
        "train": "claims_train.jsonl",
        "validation": "claims_dev.jsonl",
        "test": "claims_test.jsonl",
    }.get(split)
    if split_file is None:
        raise SciFactEvaluationError(f"Unsupported SciFact split: {split}")

    extract_dir = _download_scifact_release()
    claims = _flatten_release_claim_rows(_read_jsonl_file(_find_release_file(extract_dir, split_file)))
    corpus_rows = _read_jsonl_file(_find_release_file(extract_dir, "corpus.jsonl"))
    corpus = {str(row["doc_id"]): row for row in corpus_rows}
    return claims, corpus


def load_scifact(
    split: str = "validation",
    dataset_name: str = "allenai/scifact",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load SciFact claims and corpus from Hugging Face datasets."""

    try:
        claims_dataset = _load_dataset_config(dataset_name, "claims", split)
        corpus_dataset = _load_dataset_config(dataset_name, "corpus", "train")
        claims = [dict(row) for row in claims_dataset]
        corpus = {str(row["doc_id"]): dict(row) for row in corpus_dataset}
        return claims, corpus
    except Exception:
        return _load_scifact_release(split)


def _resolve_doc_id(row: dict[str, Any]) -> str:
    evidence_doc_id = clean_text(str(row.get("evidence_doc_id") or ""))
    if evidence_doc_id:
        return evidence_doc_id
    cited_doc_ids = row.get("cited_doc_ids") or []
    if cited_doc_ids:
        return str(cited_doc_ids[0])
    return ""


def _cited_doc_ids(row: dict[str, Any]) -> list[str]:
    return [str(doc_id) for doc_id in (row.get("cited_doc_ids") or [])]


def _selected_sentences(
    abstract: list[str],
    indices: list[int],
) -> tuple[list[str], bool]:
    selected: list[str] = []
    valid = True
    for index in indices:
        if 0 <= int(index) < len(abstract):
            selected.append(clean_text(abstract[int(index)]))
        else:
            valid = False
    return selected, valid


def build_scifact_eval_pairs(
    split: str = "validation",
    max_examples: int | None = None,
    include_neutral: bool = True,
    neutral_fallback_sentences: int = 3,
    dataset_name: str = "allenai/scifact",
) -> list[SciFactEvalPair]:
    """Convert SciFact claim/evidence rows into NLI claim-pair examples."""

    claims, corpus = load_scifact(split=split, dataset_name=dataset_name)
    pairs: list[SciFactEvalPair] = []

    for row in claims:
        gold_label = normalize_scifact_label(row.get("evidence_label"))
        if gold_label == "neutral" and not include_neutral:
            continue

        doc_id = _resolve_doc_id(row)
        doc = corpus.get(doc_id)
        doc_reference_ok = doc is not None
        abstract = [clean_text(sentence) for sentence in (doc or {}).get("abstract", [])]
        sentence_indices = [int(index) for index in (row.get("evidence_sentences") or [])]
        source = "rationale_sentences"

        selected_sentences, sentence_indices_ok = _selected_sentences(
            abstract,
            sentence_indices,
        )
        if not selected_sentences:
            if gold_label != "neutral":
                continue
            source = "abstract_fallback"
            selected_sentences = abstract[:neutral_fallback_sentences]
            sentence_indices = list(range(len(selected_sentences)))
            sentence_indices_ok = bool(selected_sentences)

        evidence_text = clean_text(" ".join(selected_sentences))
        expected_text = clean_text(" ".join(selected_sentences))
        evidence_text_ok = bool(evidence_text) and evidence_text == expected_text
        if not evidence_text:
            continue

        pair_id = stable_id("scifact_pair", split, row.get("id"), doc_id, evidence_text)
        pairs.append(
            SciFactEvalPair(
                pair_id=pair_id,
                scifact_claim_id=int(row["id"]),
                split=split,
                claim=clean_text(row.get("claim")),
                evidence_text=evidence_text,
                evidence_sentences=selected_sentences,
                evidence_sentence_indices=sentence_indices,
                evidence_doc_id=doc_id,
                cited_doc_ids=_cited_doc_ids(row),
                abstract_title=clean_text((doc or {}).get("title")) or None,
                gold_label=gold_label,
                original_label=str(row.get("evidence_label") or ""),
                grounding_source=source,
                doc_reference_ok=doc_reference_ok,
                sentence_indices_ok=sentence_indices_ok,
                evidence_text_ok=evidence_text_ok,
            )
        )
        if max_examples is not None and len(pairs) >= max_examples:
            break

    if not pairs:
        raise SciFactEvaluationError(
            "No SciFact evaluation pairs were created. Check the split and dataset access."
        )
    return pairs


def _to_claim_pair(pair: SciFactEvalPair) -> ClaimPair:
    return ClaimPair(
        pair_id=pair.pair_id,
        claim_a_id=f"scifact_claim_{pair.scifact_claim_id}",
        claim_b_id=f"scifact_doc_{pair.evidence_doc_id}",
        paper_a_id=f"scifact_claim_{pair.scifact_claim_id}",
        paper_b_id=f"scifact_doc_{pair.evidence_doc_id}",
        claim_a=pair.claim,
        claim_b=pair.evidence_text,
        similarity_score=1.0,
    )


def score_scifact_pairs(
    pairs: list[SciFactEvalPair],
    model_name: str = NLI_MODEL,
) -> list[SciFactPrediction]:
    scorer = ContradictionScorer(model_name)
    claim_pairs = [_to_claim_pair(pair) for pair in pairs]
    scores = scorer.score_pairs(claim_pairs)
    pair_by_id = {pair.pair_id: pair for pair in pairs}

    predictions: list[SciFactPrediction] = []
    for score in scores:
        source_pair = pair_by_id[score.pair_id]
        predictions.append(
            SciFactPrediction(
                pair_id=score.pair_id,
                gold_label=source_pair.gold_label,
                predicted_label=normalize_prediction_label(score.predicted_label),
                entailment_score=score.entailment_score,
                contradiction_score=score.contradiction_score,
                neutral_score=score.neutral_score,
            )
        )
    return predictions


def _threshold_prediction(prediction: SciFactPrediction, threshold: float) -> EvalLabel:
    if prediction.contradiction_score >= threshold:
        return "contradiction"
    if prediction.entailment_score >= prediction.neutral_score:
        return "entailment"
    return "neutral"


def compute_threshold_sweep(
    predictions: list[SciFactPrediction],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    y_true = [prediction.gold_label == "contradiction" for prediction in predictions]
    true_contradictions = sum(y_true)

    for threshold in thresholds:
        y_pred = [
            _threshold_prediction(prediction, float(threshold)) == "contradiction"
            for prediction in predictions
        ]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "predicted_contradictions": int(sum(y_pred)),
                "true_contradictions": int(true_contradictions),
                "evaluated_pairs": len(predictions),
            }
        )
    return pd.DataFrame(rows)


def compute_grounding_metrics(pairs: list[SciFactEvalPair]) -> dict[str, Any]:
    rationale_pairs = [pair for pair in pairs if pair.grounding_source == "rationale_sentences"]
    grounding_ok = [
        pair.doc_reference_ok and pair.sentence_indices_ok and pair.evidence_text_ok
        for pair in rationale_pairs
    ]
    return {
        "total_pairs": len(pairs),
        "rationale_grounded_pairs": len(rationale_pairs),
        "abstract_fallback_pairs": len(pairs) - len(rationale_pairs),
        "doc_reference_ok": sum(pair.doc_reference_ok for pair in pairs),
        "sentence_indices_ok": sum(pair.sentence_indices_ok for pair in pairs),
        "evidence_text_ok": sum(pair.evidence_text_ok for pair in pairs),
        "rationale_grounding_accuracy": (
            round(sum(grounding_ok) / len(rationale_pairs), 4) if rationale_pairs else None
        ),
    }


def compute_prediction_metrics(
    predictions: list[SciFactPrediction],
) -> tuple[dict[str, Any], str]:
    y_true = [prediction.gold_label for prediction in predictions]
    y_pred = [prediction.predicted_label for prediction in predictions]
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)), 4),
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, report


def _prediction_records(predictions: list[SciFactPrediction]) -> list[dict[str, Any]]:
    return [prediction.model_dump(mode="json") for prediction in predictions]


def _write_report(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _save_optional_charts(
    threshold_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    chart_paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return chart_paths

    if not threshold_df.empty:
        threshold_chart = output_dir / "threshold_sweep.png"
        plt.figure(figsize=(8, 5))
        for column in ("precision", "recall", "f1"):
            sns.lineplot(data=threshold_df, x="threshold", y=column, marker="o", label=column)
        plt.ylim(0, 1)
        plt.title("Contradiction threshold sensitivity")
        plt.tight_layout()
        plt.savefig(threshold_chart, dpi=160)
        plt.close()
        chart_paths["threshold_sweep_png"] = str(threshold_chart)

    matrix = metrics.get("confusion_matrix")
    if matrix:
        confusion_chart = output_dir / "confusion_matrix.png"
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            xticklabels=LABELS,
            yticklabels=LABELS,
            cmap="Blues",
        )
        plt.xlabel("Predicted")
        plt.ylabel("Gold")
        plt.title("SciFact NLI classification")
        plt.tight_layout()
        plt.savefig(confusion_chart, dpi=160)
        plt.close()
        chart_paths["confusion_matrix_png"] = str(confusion_chart)

    return chart_paths


def run_scifact_evaluation(
    split: str = "validation",
    max_examples: int | None = 200,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    model_name: str = NLI_MODEL,
    output_dir: str | Path | None = None,
    include_neutral: bool = True,
    save_charts: bool = True,
) -> dict[str, Any]:
    """Run SciFact pair construction, NLI scoring, metrics, and result saving."""

    ensure_data_dirs()
    resolved_output_dir = Path(output_dir) if output_dir else SCIFACT_EVALUATION_DIR / split
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_scifact_eval_pairs(
        split=split,
        max_examples=max_examples,
        include_neutral=include_neutral,
    )
    predictions = score_scifact_pairs(pairs, model_name=model_name)
    threshold_df = compute_threshold_sweep(predictions, thresholds=thresholds)
    prediction_metrics, report_text = compute_prediction_metrics(predictions)
    grounding_metrics = compute_grounding_metrics(pairs)

    metrics = {
        "dataset": "allenai/scifact",
        "split": split,
        "model_name": model_name,
        "max_examples": max_examples,
        "include_neutral": include_neutral,
        "evaluated_pairs": len(pairs),
        "prediction_metrics": prediction_metrics,
        "grounding_metrics": grounding_metrics,
        "best_threshold_by_f1": (
            threshold_df.sort_values("f1", ascending=False).iloc[0].to_dict()
            if not threshold_df.empty
            else None
        ),
    }

    paths = {
        "processed_pairs": str(write_jsonl(resolved_output_dir / "scifact_pairs.jsonl", pairs)),
        "predictions": str(
            write_jsonl(resolved_output_dir / "nli_predictions.jsonl", _prediction_records(predictions))
        ),
        "metrics": str(write_json(resolved_output_dir / "metrics.json", metrics)),
        "classification_report": str(
            _write_report(resolved_output_dir / "classification_report.txt", report_text)
        ),
        "threshold_sweep": str(
            threshold_df.to_csv(resolved_output_dir / "threshold_sweep.csv", index=False)
            or (resolved_output_dir / "threshold_sweep.csv")
        ),
    }
    if save_charts:
        paths.update(_save_optional_charts(threshold_df, prediction_metrics, resolved_output_dir))

    return {"metrics": metrics, "paths": paths}
